from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import anderson, norm
import multiprocessing
from multiprocessing import Pool
import os
import warnings
warnings.filterwarnings("ignore")



#Setting up workers to be able to run on multiple cores
def worker(args):
    import tempfile
    #A directory for each worker to store compiled data to prevent multiple workers trying to write in the same file 
    #and casuing errors of not being able to compile that group
    #and if folder exist to use that and the last line to make sure pytensor and windows work good with paths
    worker_dir = os.path.join(tempfile.gettempdir(), f"pytensor_{os.getpid()}")
    os.makedirs(worker_dir, exist_ok=True)
    worker_dir_fwd = worker_dir.replace("\\", "/")
    os.environ["PYTENSOR_FLAGS"] = f"compiledir={worker_dir_fwd}"

    #Unpacking arguments that are stored as a tuple
    #Opening fails and processing them using fit group function  returining the data or error notification 
    #depending on the outcome
    group_nr, galaxs_path, galagr_path, chains = args
    try:
        with fits.open(galaxs_path) as galaxs, fits.open(galagr_path) as galagr:
            return fitmass(group_nr, galaxs, galagr, chains)
    except Exception as e:
        print(f"[Group {group_nr}] FAILED: {e}")
        return {"group_nr": group_nr, "failed": True, "error": str(e)}
    

#Main function working on the separating the data and modeling the masses 
def fitmass(group_nr, galaxs, galagr, chains):
    import pymc as pm
    id_group_all = np.array(galaxs[1].data["id_group"])
    mask = (id_group_all == group_nr)
    # Galaxy data for hte group from galaxygroup file
    zgroup = galaxs[1].data["z"][mask] * 299792.458
    ra_gals = galaxs[1].data["ra"][mask]
    dec_gals = galaxs[1].data["dec"][mask]
    n_gals      = len(zgroup)
    
    #Galaxy group centre data from galaxy group file
    gr_mask = (galagr[1].data["id_group"] == group_nr)
    centre_dist = galagr[1].data["dist"][gr_mask][0]
    centre_ra = galagr[1].data["ra"][gr_mask][0]
    centre_dec = galagr[1].data["dec"][gr_mask][0]
    #How far is every galaxy from the group centre
    ra_per_gals=(ra_gals-centre_ra)*np.cos(np.radians(dec_gals))
    dec_per_gals=(dec_gals-centre_dec)
    angle_for_rsky = np.radians(np.sqrt(ra_per_gals**2+dec_per_gals**2))
    #rskygroup=centre_dist*np.radians(np.sqrt(ra_per_gals**2+dec_per_gals**2))
    #Model responsible for the mass estimation
    with pm.Model() as massmodel:
        # distprior = pm.Normal('distprior', mu=np.mean(centre_dist), sigma=200)
        # distsigma = pm.HalfNormal('distsigma', sigma=400)
        distance=pm.Normal("dist_likelihood", mu=centre_dist, sigma=200)



        #Setting up prioirs for the velocity
        #Normal distribution for mean velocity because we expect it to be similar to that
        #and halfnormal because the dispersion cannot be negative
        vprior = pm.Normal('vprior', mu=np.mean(zgroup), sigma=500)
        vsigma = pm.HalfNormal('vsigma', sigma=500)
        
        #Likelihood for the velocity
        #and halfnormal for the distance from the centre cannot be neg
        #weibull because connects two normal distributions in a plane
        pm.Normal("v_likelihood", mu=vprior, sigma=vsigma, observed=zgroup)

        skysigma = pm.HalfNormal('skysigma', sigma=4)
        theta_sigma = skysigma / distance
        beta_weibull = theta_sigma * np.sqrt(2)
        pm.Weibull("sky_likelihood",alpha=2, beta=beta_weibull, observed=angle_for_rsky)
        #Determinstic distribution because we calculate and connect it from previous data and it is not a random variable
        Rg=pm.Deterministic('Rg', 4.5882*skysigma)
        mass_virial1 = pm.Deterministic("mass_virial", 2.325e2*Rg * (vsigma/100)**2)
        #Sampling the data and countinf divergences, one core so it doesnt spawn child processes
        idata = pm.sample(draws=1000, tune=1000, cores=1, chains=chains, target_accept=0.95, progressbar=False, return_inferencedata=True)
        n_div = int(idata.sample_stats.diverging.sum())
    # Defining the percentiles function, 50 is mean 16 and 84 are error bars
    def percentiles(samples):
        p16, p50, p84 = np.percentile(samples, [16, 50, 84])
        return p16, p50, p84, (p84 - p50), (p50 - p16)
# Posterior to list from multi dim array
    v_samples = idata.posterior["vsigma"].values.flatten()
    s_samples = idata.posterior["skysigma"].values.flatten()
    m_samples = idata.posterior["mass_virial"].values.flatten()

    #Calculating the percentiles
    v16, v50, v84, v_ep, v_em = percentiles(v_samples)
    s16, s50, s84, s_ep, s_em = percentiles(s_samples)
    m16, m50, m84, m_ep, m_em = percentiles(m_samples)
    #Printing out data to have overview
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Group {group_nr}  |  {n_gals} galaxies")
    print(sep)
    #returns a dictionary of data per group
    return {
        "group_nr" : group_nr,
        "n_gals"   : n_gals,
        "failed"   : False,
        "divergences": n_div,
        # medians
        "v50": v50, "s50": s50, "m50": m50,
        # upper errors
        "v_ep": v_ep, "s_ep": s_ep, "m_ep": m_ep,
        # lower errors
        "v_em": v_em, "s_em": s_em, "m_em": m_em,
        # raw quantiles
        "v16": v16, "v84": v84,
        "s16": s16, "s84": s84,
        "m16": m16, "m84": m84,
    }


def main():
    galaxs_path = "SDSSgalaxies.fits"
    galagr_path = "SDSSgroups.fits"
    #user input how may galaxies to process and compute in parallel and how many chains to run per group
    group_count = int(input("\nHow many groups to process?          : "))
    num_workers = int(input("Groups to run in parallel (workers)? : "))
    chains      = int(input("How many MCMC chains per group?      : "))
    processed_groups=set()

    if os.path.exists("results_summary.txt"):
        with open("results_summary.txt", "r") as f:
            for line in f:
                try:
                    line=line.split()
                    processed_groups.add(int(line[0]))
                except (ValueError, IndexError):
                    pass
    #all necessary arguments for workers because pool takes only one argument
    args_list = [
        (group_nr, galaxs_path, galagr_path, chains)
        for group_nr in range(1, group_count + 1)
        if group_nr not in processed_groups
    ]
    #Starting time
    t0 = time.time()
    computed_results = []
    batch_size=1000

    if not os.path.exists("results_summary.txt"):
        with open("results_summary.txt", "w") as f:
            f.write(f"{'Group':>6}  {'N':>4}  "
                f"{'v50':>8}  {'v_ep':>7}  {'v_em':>7}  "
                f"{'s50':>8}  "
                f"{'m50':>10}  {'m_ep':>9}  {'m_em':>9} {'div':>4}\n")
    # setting up multiprocessing and saves the results in raw results
    mp_context = multiprocessing.get_context("spawn")
    with mp_context.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(worker, args_list):
            if result and not result.get("failed"):
                computed_results.append(result)
            else:
                pass
            if len(computed_results) >= batch_size:
                with open("results_summary.txt", "a") as f:
                    for elem in computed_results:
                        f.write(f"{elem['group_nr']:>6}  {elem['n_gals']:>4}  {elem['v50']:>8.2f}  {elem['v_ep']:>7.2f}  {elem['v_em']:>7.2f}  {elem['s50']:>8.3f}  {elem['m50']:>10.2f}  {elem['m_ep']:>9.2f}  {elem['m_em']:>9.2f}  {elem['divergences']:>4}\n")
                computed_results = []
        if computed_results!= []:
            with open("results_summary.txt", "a") as f:
                for elem in computed_results:
                    f.write(f"{elem['group_nr']:>6}  {elem['n_gals']:>4}  {elem['v50']:>8.2f}  {elem['v_ep']:>7.2f}  {elem['v_em']:>7.2f}  {elem['s50']:>8.3f}  {elem['m50']:>10.2f}  {elem['m_ep']:>9.2f}  {elem['m_em']:>9.2f}  {elem['divergences']:>4}\n")
            computed_results = []
    elapsed = time.time() - t0
    print(f"\nAll groups finished in {elapsed:.1f} s")
 
if __name__ == '__main__':
    main()
