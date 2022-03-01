from ROOT import TLorentzVector


class Particle(TLorentzVector):
    def __init__(self, pt, eta, phi, mass):
        TLorentzVector.__init__(self)
        self.SetPtEtaPhiM(pt, eta, phi, mass)
        self.charge = 0.
        self.is_muon = False
        self.is_electron = False
        self.is_jet = False
        self.btagScore = 0.

    def IsMuon(self):
        return self.is_muon

    def IsElectron(self):
        return self.is_electron

    def IsJet(self):
        return self.is_jet

    def Charge(self):
        return self.charge

    def BtagScore(self):
        return self.btagScore


class Lepton(Particle):
    # LeptonType
    # 1: ewprompt
    # 2: signal muons, from A
    # 3: muons from tau
    # 6: from offshell W, i.e. directly from Hc
    # <0: fake leptons
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)
        self.is_muon = True

    def SetCharge(self, charge):
        self.charge = charge

    def SetLepType(self, lepType):
        self.lepType = lepType

    def SetMiniIso(self, miniIso):
        self.miniIso = miniIso

    def SetIsTight(self, isTight):
        self.isTight = isTight

    def LepType(self):
        return self.lepType

    def MiniIso(self):
        return self.miniIso

    def IsTight(self):
        return self.isTight


class Jet(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)
        self.is_bjet = False

    def SetBtagScore(self, btagScore):
        self.btagScore = btagScore

    def BtagScore(self):
        return self.btagScore

    def SetIsBtagged(self, isBtagged):
        self.isBtagged = isBtagged

    def isBtagged(self):
        return self.isBtagged


# Useful functions
def get_leptons(evt):
    muons = []
    muons_zip = zip(evt.muons_pt, evt.muons_eta, evt.muons_phi, evt.muons_mass,
                    evt.muons_charge, evt.muons_lepType, evt.muons_miniIso,
                    evt.muons_isTight)
    for pt, eta, phi, mass, charge, lepType, miniIso, isTight in muons_zip:
        this_muon = Lepton(pt, eta, phi, mass)
        this_muon.SetCharge(charge)
        this_muon.SetLepType(lepType)
        this_muon.SetMiniIso(miniIso)
        this_muon.SetIsTight(isTight)
        muons.append(this_muon)
    # check the number of muons
    if evt.nMuons != len(muons):
        Warning(f"muon entry is different, {evt.nMuons}: {len(muons)}")

    electrons = []
    electrons_zip = zip(
        evt.electrons_pt,
        evt.electrons_eta,
        evt.electrons_phi,
        evt.electrons_mass,
        evt.electrons_charge,
        evt.electrons_lepType,
        evt.electrons_miniIso,
        evt.electrons_isTight,
    )
    for pt, eta, phi, mass, charge, lepType, miniIso, isTight in electrons_zip:
        this_electron = Lepton(pt, eta, phi, mass)
        this_electron.SetCharge(0)
        this_electron.SetLepType(lepType)
        this_electron.SetMiniIso(miniIso)
        this_electron.SetIsTight(isTight)
        electrons.append(this_electron)
    # check the number of electrons
    if evt.nElectrons != len(electrons):
        Warning(
            f"electron entry is different, {evt.nElectrons}: {len(electrons)}")

    return muons, electrons


def get_jets(evt):
    jets = []
    jets_zip = zip(
        evt.jets_pt,
        evt.jets_eta,
        evt.jets_phi,
        evt.jets_mass,
        evt.jets_btagScore,
        evt.jets_isBtagged,
    )
    for pt, eta, phi, mass, btagScore, isBtagged in jets_zip:
        this_jet = Jet(pt, eta, phi, mass)
        this_jet.SetBtagScore(btagScore)
        this_jet.SetIsBtagged(isBtagged)
        jets.append(this_jet)
    # check the number of jets
    if evt.nJets != len(jets):
        Warning(f"jet entry is different, {evt.nJets}: {len(jets)}")

    bjets = []
    for jet in jets:
        if jet.isBtagged:
            bjets.append(jet)

    return jets, bjets