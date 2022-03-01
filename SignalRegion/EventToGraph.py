import random
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from Scripts.DataFormat import Particle, get_leptons, get_jets


def select_loosen(channel, evt, muons, electrons, jets):
    # 1E2Mu
    # 1. Should pass triggers and safe cuts
    # 2. 1E2Mu
    # 3. Exist OS muon pair with M(OSSF) > 12 GeV
    # 4. Nj >= 2, Nb >= 1
    if channel == "1E2Mu":
        if not (evt.passDblMuTrigs or evt.passEMuTrigs):
            return False
        pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.)
                        or (muons[0].Pt() > 25. and electrons[0].Pt() > 15.)
                        or (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
        if not pass_safecut:
            return False

        if not len(jets) >= 2:
            return False

        return True

    elif channel == "3Mu":
        raise (NotImplementedError)
    else:
        raise (KeyError)


def get_prompt_leptons(muons, electrons):
    muons_prompt = []
    electrons_prompt = []
    for muon in muons:
        if muon.LepType() < 0: continue
        muons_prompt.append(muon)
    for electron in electrons:
        if electron.LepType() < 0.: continue
        electrons_prompt.append(electron)

    return muons_prompt, electrons_prompt


class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(MyDataset, self).__init__("./tmp/MyDataset")
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

    #@property
    #def raw_file_names(self):
    #    return ['file']

    #@property
    #def processed_file_names(self):
    #    return ['data.pt']


def get_edge_indices(node_list, k=4):
    edge_index = []
    edge_attribute = []
    for i, node in enumerate(node_list):
        distances = {}  # apply K-NN edge with deltaR
        for j, neighbor in enumerate(node_list):
            if i == j:  # same node
                continue
            deta = node[1] - neighbor[1]
            dphi = np.remainder(node[2] - neighbor[2], np.pi)
            distances[j] = np.sqrt(np.power(deta, 2) + np.power(dphi, 2))
            sorted_dRs = dict(
                sorted(distances.items(), key=lambda item: item[1]))
            for n in list(sorted_dRs.keys())[:k]:
                edge_index.append([i, n])
                edge_attribute.append([distances[n]])

    return torch.tensor(edge_index,
                        dtype=torch.long), torch.tensor(edge_attribute,
                                                        dtype=torch.float)


# Get objects(nodes) of the first event(graph)
def rtfile_to_datalist(rtfile, is_signal=False):
    data_list = []
    for evt in rtfile.Events:
        muons, electrons = get_leptons(evt)
        jets, bjets = get_jets(evt)
        METv = Particle(evt.METv_pt, 0., evt.METv_phi, 0.)

        if not select_loosen("1E2Mu", evt, muons, electrons, jets):
            continue

        muons_prompt, electrons_prompt = get_prompt_leptons(muons, electrons)
        is_prompt_evt = (len(muons) == len(muons_prompt)
                         and len(electrons) == len(electrons_prompt))
        if not is_signal and not is_prompt_evt: pass  # fake events
        elif is_signal and is_prompt_evt: pass  # signal events
        else: continue  # to reduce noise

        # Now let's convert the event to a single graph
        node_list = []
        for particle in muons + electrons + jets:
            node_list.append([
                particle.Pt(),
                particle.Eta(),
                particle.Phi(),
                particle.M(),
                particle.Charge(),
                particle.IsMuon(),
                particle.IsElectron(),
                particle.IsJet(),
                particle.BtagScore()
            ])

        x = torch.tensor(node_list, dtype=torch.float)

        # make edges
        # NOTE: It is directed graph!
        # for each node, find 3 nearest particles and connect them
        edge_index, edge_attribute = get_edge_indices(node_list, k=4)
        data = Data(x=x,
                    y=int(is_signal),
                    edge_index=edge_index.t().contiguous(),
                    edge_attribute=edge_attribute)
        data_list.append(data)
        if len(data_list) == 15000:
            break

    return data_list
