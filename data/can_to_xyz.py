from __future__ import print_function
from itertools import combinations
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent import futures
import progressbar
import sys
def remove_confs(mol, energy, rms):
    e = []
    for conf in mol.GetConformers():
        #ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())

        if ff is None:
            print(Chem.MolToSmiles(mol))
            return
        e.append((conf.GetId(), ff.CalcEnergy()))
    e = sorted(e, key=lambda x: x[1])

    if not e:
        return

    kept_ids = [e[0][0]]
    remove_ids = []

    for item in e[1:]:
        if item[1] - e[0][1] <= energy:
            kept_ids.append(item[0])
        else:
            remove_ids.append(item[0])

    if rms is not None:
        rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(kept_ids, 2)]
        while any(item[2] < rms for item in rms_list):
            for item in rms_list:
                if item[2] < rms:
                    i = item[1]
                    remove_ids.append(i)
                    break
            rms_list = [item for item in rms_list if item[0] != i and item[1] != i]
    for cid in set(remove_ids):
        mol.RemoveConformer(cid)
# 重新排列分子的构象，保证构象加入的顺序与构象id从小到大的顺序一致
def reorder_confs(mol):
    """
    Reorder conformers in a molecule
    重新排列分子的构象，改变构象id以保证构象加入的顺序与构象id从小到大的顺序一致

    Returns
    -------
    mol

    """
    confs = mol.GetConformers()
    confs = sorted(confs, key=lambda x: x.GetId())
    for i, conf in enumerate(confs):
        conf.SetId(i)
    return mol
# 生成单个分子的构象
def gen_confs_mol(mol, nconf=5, energy=100, rms=0.5, seed=42):
    """
    Generates conformations for a single molecule

    Returns
    -------
    mol:Mol with conformations

    """
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=700, randomSeed=seed)

    for cid in cids:
        try:
            #AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        except:
            continue
    remove_confs(mol, energy, rms)
    mol = reorder_confs(mol)

    return mol
# This function is called in the subprocess.
def generateconformations(m):
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    m = Chem.RemoveHs(m)
    return m
if __name__ == "__main__":
    max_workers=12

    data = sys.argv[1]
    smiles_name = data+'.can'
    xyz_name = data+'.xyz'

    #n = int(sys.argv[2])

    writer = open(xyz_name, "w")

    suppl = Chem.SmilesMolSupplier(smiles_name, delimiter="\t", titleLine=True)

    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit a set of asynchronous jobs
        jobs = []
        for mol in suppl:
            if mol:
                job = executor.submit(gen_confs_mol, mol, nconf=5, energy=100, rms=0.5, seed=42)
                jobs.append(job)

        widgets = ["Generating conformations; ", progressbar.Percentage(), " ",
                progressbar.ETA(), " ", progressbar.Bar()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(jobs))

        # Process the job results (in submission order) and save the conformers.
        for job in jobs:
            mol= job.result()
            for conf in mol.GetConformers():
                writer.write(Chem.MolToXYZBlock(mol, confId=conf.GetId()))
            
    writer.close()
