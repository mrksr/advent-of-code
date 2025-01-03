def simple() -> tuple[int, str]:
    return (
        2,
        2,
        r"""RL

    AAA = (BBB, CCC)
    BBB = (DDD, EEE)
    CCC = (ZZZ, GGG)
    DDD = (DDD, DDD)
    EEE = (EEE, EEE)
    GGG = (GGG, GGG)
    ZZZ = (ZZZ, ZZZ)
    """,
    )


def multi_step() -> tuple[int, str]:
    return (
        6,
        6,
        r"""LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)
""",
    )


def parallel() -> tuple[int, str]:
    return (
        2,
        6,
        r"""LR

AAA = (11B, XXX)
11B = (XXX, ZZZ)
ZZZ = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)
""",
    )


def full() -> tuple[int, str]:
    return (
        15517,
        42,
        r"""LRRRLRLLLLLLLRLRLRRLRRRLRRLRRRLRRLRRRLLRRRLRRLRLRRRLRRLRRRLLRLLRRRLRRRLRLLRLRLRRRLRRLRRLRRLRLRRRLRRLRRRLLRLLRLLRRLRLLRLRRLRLRLRRLRRRLLLRRLRRRLLRRLRLRLRRRLRLRRRLLRLLLRRRLLLRRLLRLLRRLLRLRRRLRLRRLRRLLRRLRLLRLRRRLRRRLRLRRRLRLRLRRLRLRRRLRRRLRRRLRRLRRLRRRLLRLRLLRLLRRRR

HVX = (SCS, XQN)
DMK = (JKL, JKL)
FDF = (XHL, RMM)
JTK = (SVN, DVP)
QCF = (FCH, FCH)
TCG = (VMS, SDL)
JJP = (FQJ, RLT)
DRP = (RMJ, RMJ)
VKF = (XQB, VBX)
HRS = (BXK, DPM)
FHH = (FBF, TNX)
HDJ = (MBH, QRM)
TMG = (LJJ, JVM)
KJK = (GXP, FHS)
LKV = (VCV, JDP)
CVT = (MBM, BSQ)
RSD = (BFH, BPP)
KJG = (TMB, DMC)
DRH = (BFS, NCJ)
GRF = (MJL, TXX)
JVA = (XKG, RCL)
GFK = (NJF, NPR)
CQJ = (GVM, KKF)
TJC = (FNR, TFH)
BJP = (NMJ, JMX)
DRX = (MRR, QDL)
QFV = (TLM, XQM)
XQF = (NGF, NDC)
GDF = (TCV, PTP)
RHQ = (RGT, PXT)
NBR = (RRV, NCG)
PVF = (QPL, KRG)
CSQ = (TSC, GRF)
VMJ = (VTK, HRS)
BKH = (RPF, JGX)
GMM = (SRN, MSQ)
NLK = (BHP, TVR)
JDS = (LBV, TBN)
FQJ = (FMG, NRV)
JNG = (LTX, KHH)
DNM = (QPL, KRG)
BCJ = (LXV, BKQ)
LHD = (SGJ, JBX)
NMJ = (GPT, BCL)
DNJ = (DFT, BXT)
RDX = (QCX, VBF)
VMS = (HGR, MLS)
CSR = (PMC, BPT)
LFT = (FXN, SRJ)
JRK = (GDC, KVK)
BMB = (LJJ, JVM)
JMD = (JXX, JDS)
GPS = (LQF, QXR)
GQF = (VFK, GDR)
GGB = (MBP, RMV)
NRR = (CDX, MDJ)
VDM = (QRJ, LDJ)
FHM = (GFK, DSM)
XQM = (QSH, SXJ)
PSG = (RMP, CLT)
SPH = (VQQ, QTG)
XDB = (NSX, HDJ)
GDC = (SBL, XCL)
QBH = (HGS, KMR)
JDK = (PXR, LFT)
VBX = (DPL, VNL)
GXQ = (LFP, BXD)
RTT = (DHV, GDM)
RRV = (KRF, PLS)
XLA = (XMF, TRG)
LTP = (FNR, TFH)
SRP = (SNR, DLD)
JVV = (BCN, QXH)
JKM = (VCS, SQB)
DXX = (SDN, VXM)
XCV = (JBX, SGJ)
VXL = (XJM, HKT)
DGB = (RDQ, HGM)
SLT = (GDL, NQV)
XHH = (PQN, DNP)
FBK = (GMM, RHM)
PCG = (TCV, PTP)
GPT = (BCJ, NQH)
RLK = (TMG, BMB)
DSJ = (JKP, PKN)
XXJ = (VHX, RNJ)
LPS = (FBQ, NFG)
TMS = (KTV, VLT)
FXN = (CLC, HJJ)
GSV = (XSM, PPQ)
PQN = (XXJ, FJP)
HKX = (JSQ, RFS)
TRS = (BRL, FLC)
CPK = (QVN, PRR)
VFK = (KRH, LFV)
XVJ = (MQK, LRG)
BXT = (DMQ, JMD)
CDH = (VQJ, CLR)
FLP = (VPN, VBT)
KDF = (LPD, KMS)
HXH = (DNQ, CDL)
LLH = (SJD, JXB)
PKN = (KBD, RXT)
MJT = (FCH, CGS)
HMN = (PNK, QRK)
TXR = (QFV, DJM)
KNF = (HST, SLT)
XHL = (RXM, HXF)
VCM = (QBH, QHS)
NXD = (CXH, TKV)
BPP = (LBD, TBR)
TKX = (LPQ, JNG)
SVM = (LHN, PVJ)
BGH = (KNS, VBG)
HHX = (QJP, QMC)
CCN = (GSV, PHX)
TNH = (TRP, JXT)
LJM = (PMV, BGR)
TJX = (XNM, GBF)
LCD = (XQF, KRP)
PTP = (CSR, SDG)
RCL = (BMJ, RMH)
DVQ = (JCJ, SCX)
DJM = (XQM, TLM)
TVR = (VLX, XQZ)
KXH = (DHV, GDM)
HFB = (NQB, LKZ)
LBV = (MDV, LKP)
XMF = (DXS, JRM)
DSG = (XFF, HMN)
VNL = (FQD, RRC)
DPB = (TRP, JXT)
HHR = (DVT, DTJ)
CLR = (NXD, XPH)
QSF = (FTV, LLH)
JRM = (LPS, VGL)
QHK = (DRP, DRP)
GKJ = (QQN, JGC)
BKQ = (FBK, RJX)
VBG = (GCV, DHH)
JGC = (NFF, RHQ)
KBD = (TFJ, XBN)
QHG = (JMQ, XGL)
DFJ = (MMB, DRX)
HLJ = (GHG, HXG)
KKF = (TKX, XTM)
LXP = (GBF, XNM)
KKG = (TJX, LXP)
BGR = (PFQ, FSG)
BTS = (BCN, QXH)
BXK = (VKL, FSX)
JBS = (PXR, LFT)
KHF = (GBD, FCP)
CLK = (FHM, GVG)
NPR = (JTM, VDR)
XGL = (DMK, JHG)
SFV = (KXH, RTT)
DSM = (NJF, NPR)
DKH = (BSQ, MBM)
FTX = (LXP, TJX)
PLF = (DDS, LXJ)
JTB = (HLJ, DGS)
TJB = (QHG, LDQ)
RFC = (NPF, BXL)
MBH = (KFQ, DMT)
LCM = (DGV, HKX)
DLM = (GQD, KGH)
GSM = (QDQ, RHN)
PKV = (DTJ, DVT)
HNC = (PHX, GSV)
JBT = (DSR, DSL)
RMV = (XMT, GND)
QRK = (DFH, FFT)
QDQ = (RJC, FTF)
MFX = (DPB, TNH)
LGS = (JVF, JRK)
JFR = (QLV, JQN)
BXL = (LQH, JFR)
GDM = (DTB, GXR)
PTG = (XVJ, MNT)
HGS = (CGM, LSS)
LDQ = (JMQ, XGL)
LPR = (PQN, DNP)
TRG = (JRM, DXS)
VHX = (JTQ, BKP)
QPP = (LFP, BXD)
NNC = (RFQ, RDT)
PJL = (NGD, SXQ)
JQN = (PJL, CTJ)
JGH = (LRQ, JNN)
TRP = (PJR, SSS)
FTD = (HHR, PKV)
GXR = (VDL, VCM)
TXT = (HNC, CCN)
SQB = (LPR, XHH)
NLH = (QLR, LXH)
RCR = (DNQ, CDL)
CGM = (TJB, GCM)
DGS = (GHG, HXG)
TMB = (GGQ, MSB)
VJV = (PSR, BPR)
NNN = (LQF, QXR)
FNK = (PSG, HTG)
NCJ = (KJK, FSJ)
GRC = (SQP, NRR)
HQD = (LHN, PVJ)
NBL = (PNT, RSP)
JLF = (BHP, BHP)
HTG = (CLT, RMP)
QQN = (RHQ, NFF)
NRV = (LKV, FLQ)
RBV = (CMQ, GBG)
ZZZ = (SCX, JCJ)
NGF = (QFD, HVX)
XPH = (CXH, TKV)
CBR = (HTD, THC)
BCL = (NQH, BCJ)
SSS = (GMD, HBV)
GHL = (JTB, MHT)
QSH = (RRS, HGX)
CVS = (JKP, PKN)
QNK = (CVT, DKH)
JBX = (LSF, HNG)
HFQ = (DDT, QNK)
VQP = (BLP, PPF)
RGD = (BLP, PPF)
SDX = (PLX, SFL)
MLT = (XRS, LGS)
DSL = (QCF, MJT)
MFP = (HTS, GRC)
BRL = (MRS, JBT)
BJV = (HTD, THC)
PLV = (PNL, JKR)
PMV = (PFQ, FSG)
KHH = (MKL, RQF)
HCQ = (RQX, JVL)
VTD = (DNJ, KDB)
GXP = (JMF, RSD)
MGC = (DVQ, ZZZ)
CSS = (SRB, HTQ)
CTJ = (SXQ, NGD)
DNA = (RDJ, DGB)
FBH = (RGF, DRH)
DVM = (BXM, QFM)
MSB = (PVK, XKN)
XBQ = (JKR, PNL)
JXT = (PJR, SSS)
CLX = (XVJ, MNT)
DPL = (FQD, RRC)
QRJ = (CGT, KXN)
KRB = (TTV, DVM)
HGX = (RSM, GPP)
RMP = (XCV, LHD)
BXD = (CSS, FBT)
SRQ = (SXR, NBL)
JGX = (JLF, NLK)
DVP = (BKH, XXF)
SFL = (DFN, RLL)
BMF = (RHB, NTZ)
NPF = (LQH, JFR)
SMJ = (DSG, SQD)
CCH = (NCG, RRV)
KGH = (MBD, CCC)
DLS = (HMJ, PLF)
AAA = (JCJ, SCX)
XRP = (HCT, VXV)
SCS = (TLP, KKP)
RPL = (KBM, GKJ)
SVD = (NTN, BXF)
FMG = (LKV, FLQ)
XXF = (RPF, JGX)
QFJ = (VBT, VPN)
NKH = (GPS, NNN)
HST = (NQV, GDL)
CMK = (CLR, VQJ)
KTF = (LSN, RQS)
KDR = (HRD, SDX)
VLX = (XMF, TRG)
NFM = (MBP, RMV)
GRJ = (QDQ, RHN)
QHN = (TLL, GVB)
CRX = (SLT, HST)
DMS = (KGH, GQD)
QJP = (DLM, DMS)
KMR = (LSS, CGM)
FCH = (BCB, BCB)
PHX = (PPQ, XSM)
SGJ = (LSF, HNG)
TNX = (JHP, PDL)
RGT = (RXG, QRT)
DKC = (SRQ, TBK)
PVM = (CJP, VJV)
CDL = (FHT, SGR)
PBK = (PMD, QHN)
KTV = (MSX, SRP)
SQD = (XFF, HMN)
GQD = (CCC, MBD)
CFS = (VXL, MTH)
QVC = (LKS, PFT)
RQX = (NKD, RBV)
BXM = (JTP, PJD)
NKD = (CMQ, GBG)
VTK = (DPM, BXK)
SDG = (BPT, PMC)
KXR = (QNK, DDT)
FNJ = (LRQ, JNN)
RJX = (RHM, GMM)
QHS = (HGS, KMR)
GDL = (BGH, HPT)
GVM = (TKX, XTM)
DHV = (DTB, GXR)
NGD = (QMN, TCG)
NQB = (RDJ, DGB)
KRP = (NGF, NDC)
JKP = (KBD, RXT)
PGN = (VLT, KTV)
DFN = (CMK, CDH)
MHT = (HLJ, DGS)
BLP = (PCS, VMJ)
MDV = (HVV, KRB)
JXX = (LBV, TBN)
LDJ = (CGT, KXN)
NQV = (BGH, HPT)
NTZ = (FQL, JSG)
GND = (GSM, GRJ)
XHM = (LJM, TDR)
BNK = (QHK, RGP)
DMC = (GGQ, MSB)
FQL = (BJP, LXM)
CLT = (LHD, XCV)
JGD = (VXM, SDN)
XMT = (GRJ, GSM)
DFT = (JMD, DMQ)
CCC = (QFJ, FLP)
LFP = (FBT, CSS)
PMD = (GVB, TLL)
DGH = (RMJ, MGC)
TBK = (NBL, SXR)
FQD = (QBK, QBK)
RDJ = (RDQ, HGM)
FTV = (SJD, JXB)
GDR = (KRH, LFV)
GMD = (SFB, LQL)
DRV = (MPP, GFG)
TFH = (VDM, NMG)
FTF = (DFJ, CHK)
FLQ = (JDP, VCV)
JVF = (GDC, KVK)
PJD = (RVN, RVG)
XNM = (QCR, NLH)
RVG = (TXT, XVD)
TSC = (TXX, MJL)
KRF = (SVD, QJR)
VPQ = (PVF, DNM)
JHP = (JKM, DMJ)
QJK = (HCQ, VMT)
NRD = (XMH, STF)
RXF = (DVK, JFG)
RLT = (NRV, FMG)
TCV = (SDG, CSR)
XKG = (BMJ, RMH)
PXF = (JHS, GHP)
NFB = (BHJ, GLF)
FBT = (SRB, HTQ)
SRB = (GKB, TVB)
CLC = (RKB, DHT)
LKP = (HVV, KRB)
SPQ = (FHM, GVG)
MRR = (HHX, BPD)
QHQ = (BMB, TMG)
RLM = (BSF, BJH)
TBR = (KLV, XBZ)
BPR = (BSR, DKP)
MNL = (NKG, TXL)
DGV = (RFS, JSQ)
XBN = (KJG, CMV)
QMN = (VMS, SDL)
SDN = (TMS, PGN)
KMS = (FFD, MFB)
HJJ = (DHT, RKB)
HTS = (SQP, NRR)
FFT = (MMH, XXB)
KFQ = (LMH, KDF)
SHA = (JSG, FQL)
RXM = (RFC, KHD)
PCB = (TNX, FBF)
LFV = (DXC, SMJ)
XVD = (CCN, HNC)
GFG = (GMN, PNC)
RDQ = (GHL, MKP)
QCR = (QLR, LXH)
JCJ = (DLS, HMG)
GSJ = (QHN, PMD)
TLL = (GTD, BNK)
LPD = (MFB, FFD)
BPL = (GQF, JTF)
QVG = (LKS, PFT)
SQP = (CDX, MDJ)
SXR = (RSP, PNT)
XBZ = (PXF, SSF)
JDD = (DPB, TNH)
HGR = (QPP, GXQ)
PFT = (GLP, CBJ)
PNK = (FFT, DFH)
RMM = (RXM, HXF)
RXD = (BTS, JVV)
HVV = (TTV, DVM)
JMX = (GPT, BCL)
NFF = (PXT, RGT)
RHM = (SRN, MSQ)
DFH = (XXB, MMH)
PXT = (RXG, QRT)
VXM = (PGN, TMS)
FCP = (LDN, TRS)
NFG = (CFS, LGC)
RFS = (JDK, JBS)
RRC = (QBK, CCL)
KRH = (DXC, SMJ)
TKV = (VKF, HHL)
CGT = (HHF, BKT)
QLN = (FCP, GBD)
TTQ = (TDT, MQS)
CRJ = (MPP, GFG)
KDZ = (RCL, XKG)
TNC = (DVP, SVN)
TLM = (SXJ, QSH)
HCD = (VXV, HCT)
BCB = (NQB, NQB)
TXX = (LTJ, TTQ)
FJQ = (RXR, FTD)
KQG = (QFV, DJM)
JTP = (RVG, RVN)
DHH = (FKM, CKS)
GMN = (GGB, NFM)
DVK = (CMB, JJP)
MLX = (RCR, HXH)
MLS = (GXQ, QPP)
JKR = (QSF, RMR)
XRS = (JVF, JRK)
LTX = (RQF, MKL)
MBP = (GND, XMT)
MMH = (PFD, SFV)
LQH = (QLV, JQN)
KLV = (SSF, PXF)
LBD = (KLV, KLV)
LRQ = (LCM, FXB)
PLM = (GJF, FNK)
DNP = (FJP, XXJ)
PNT = (MLT, XGD)
VVS = (TBK, SRQ)
KXN = (HHF, BKT)
GTD = (QHK, RGP)
CDN = (XRP, HCD)
HNG = (DNC, VPQ)
QXR = (HBT, DCD)
FHV = (TXR, KQG)
GVB = (GTD, BNK)
KDB = (DFT, BXT)
PDH = (RFQ, RDT)
XST = (BSF, BJH)
VDL = (QBH, QHS)
RVN = (XVD, TXT)
HCT = (HQD, SVM)
VDR = (PDH, NNC)
RHN = (FTF, RJC)
BHF = (VBF, QCX)
GHX = (RHB, RHB)
CDX = (KVP, KDR)
NDC = (QFD, HVX)
GSH = (BHF, RDX)
MSQ = (KLJ, FLH)
BSR = (HPC, FBH)
SBL = (VNF, FDF)
RPF = (JLF, JLF)
LSS = (TJB, GCM)
GBG = (RXD, GSC)
JMQ = (DMK, DMK)
BKT = (PBK, GSJ)
PNL = (RMR, QSF)
LXJ = (CSQ, PHG)
JTF = (VFK, GDR)
KVK = (XCL, SBL)
MPG = (KQG, TXR)
JQC = (PCG, GDF)
JMF = (BFH, BFH)
PRR = (KLG, CDN)
PNJ = (KKF, GVM)
HHL = (XQB, VBX)
BXF = (NLF, JQC)
XQN = (TLP, KKP)
TTV = (QFM, BXM)
TXL = (KTF, QGQ)
XQZ = (TRG, XMF)
RQF = (LCD, SJC)
SCX = (DLS, HMG)
RXG = (PCB, FHH)
GSC = (JVV, BTS)
VQJ = (NXD, XPH)
JDP = (VTD, BNC)
PLX = (RLL, DFN)
FJP = (RNJ, VHX)
LHN = (XHM, NMC)
SGR = (MXT, QPR)
FKM = (JTK, TNC)
QXH = (HFQ, KXR)
RGF = (NCJ, BFS)
XKN = (NTT, MLX)
BPT = (VQP, RGD)
SJC = (XQF, KRP)
CBJ = (HRM, CPK)
TLP = (NPG, RPL)
KLJ = (CVS, DSJ)
SXJ = (RRS, HGX)
LGC = (VXL, MTH)
QGQ = (RQS, LSN)
HRM = (QVN, PRR)
GCV = (CKS, FKM)
SSF = (JHS, GHP)
HMJ = (LXJ, DDS)
PXR = (FXN, SRJ)
QJR = (NTN, BXF)
TSN = (XDB, RJV)
FLC = (MRS, JBT)
LMH = (KMS, LPD)
HTD = (TJC, LTP)
VJM = (BDR, GSH)
PLS = (SVD, QJR)
LTJ = (TDT, MQS)
HBT = (VXP, MNL)
HPT = (KNS, VBG)
NCG = (KRF, PLS)
HXF = (RFC, KHD)
QDL = (HHX, BPD)
NXX = (DVK, JFG)
DXC = (SQD, DSG)
FNR = (NMG, VDM)
PVK = (NTT, MLX)
KBM = (JGC, QQN)
XSM = (KHF, QLN)
HGM = (MKP, GHL)
QBK = (GHX, GHX)
XMH = (FHV, MPG)
JNN = (LCM, FXB)
SRN = (FLH, KLJ)
JVM = (MDH, XPQ)
QFD = (SCS, XQN)
PJR = (HBV, GMD)
FXB = (HKX, DGV)
VVD = (GSH, BDR)
FFD = (BJV, CBR)
SRJ = (HJJ, CLC)
MNT = (LRG, MQK)
THC = (TJC, LTP)
JVL = (NKD, RBV)
LXM = (JMX, NMJ)
RLL = (CMK, CDH)
XJM = (PNJ, CQJ)
DSR = (QCF, MJT)
RDT = (XBQ, PLV)
VNF = (XHL, RMM)
DNC = (PVF, DNM)
BNC = (KDB, DNJ)
JTQ = (QJK, CLJ)
CKS = (TNC, JTK)
PSR = (BSR, DKP)
QFM = (JTP, PJD)
PDL = (DMJ, JKM)
FSX = (RLM, XST)
DKP = (FBH, HPC)
BCN = (HFQ, KXR)
RMH = (RLK, QHQ)
NTT = (RCR, HXH)
RGP = (DRP, DGH)
QVM = (GJF, FNK)
BSQ = (MFX, JDD)
RMJ = (DVQ, DVQ)
BDR = (BHF, RDX)
XFF = (PNK, QRK)
VLT = (MSX, SRP)
KLG = (HCD, XRP)
GLP = (HRM, CPK)
LSN = (KCQ, PVM)
RFQ = (PLV, XBQ)
LBP = (GQF, JTF)
HTQ = (GKB, TVB)
MJL = (TTQ, LTJ)
TBN = (MDV, LKP)
KCQ = (VJV, CJP)
FSJ = (GXP, FHS)
DNQ = (SGR, FHT)
VMT = (JVL, RQX)
GCM = (QHG, LDQ)
PFD = (RTT, KXH)
JSQ = (JDK, JBS)
MPP = (GMN, PNC)
LKS = (GLP, CBJ)
XGD = (XRS, LGS)
HKT = (CQJ, PNJ)
LRG = (TSN, FGK)
GVG = (DSM, GFK)
HXG = (JGD, DXX)
RHS = (GRC, HTS)
RJC = (DFJ, CHK)
MXT = (FTX, KKG)
RXR = (PKV, HHR)
NLF = (GDF, PCG)
STF = (FHV, MPG)
BFH = (LBD, LBD)
CHK = (MMB, DRX)
NPG = (KBM, GKJ)
MMB = (QDL, MRR)
LXV = (RJX, FBK)
JSG = (BJP, LXM)
LQF = (HBT, DCD)
VCV = (BNC, VTD)
PHG = (TSC, GRF)
SNR = (SPQ, CLK)
LJJ = (XPQ, MDH)
HRD = (PLX, SFL)
QLR = (MFP, RHS)
FHS = (JMF, RSD)
LQL = (CRX, KNF)
GHP = (FNJ, JGH)
NTN = (JQC, NLF)
RXT = (XBN, TFJ)
BSF = (PLJ, NKH)
BJH = (PLJ, NKH)
LPQ = (KHH, LTX)
DMQ = (JXX, JDS)
PPF = (VMJ, PCS)
PMC = (RGD, VQP)
RNJ = (BKP, JTQ)
QPR = (KKG, FTX)
VXP = (NKG, TXL)
DVT = (DRV, CRJ)
SDL = (MLS, HGR)
NQH = (LXV, BKQ)
CMQ = (RXD, GSC)
FLH = (CVS, DSJ)
FGK = (RJV, XDB)
XXB = (SFV, PFD)
MQK = (FGK, TSN)
DCD = (MNL, VXP)
DXS = (LPS, VGL)
TDT = (LTB, NRD)
BFS = (FSJ, KJK)
TDR = (PMV, BGR)
RSP = (XGD, MLT)
QMC = (DLM, DMS)
NMC = (TDR, LJM)
GBF = (NLH, QCR)
XPQ = (KMC, FJQ)
KMC = (FTD, RXR)
HBV = (SFB, LQL)
BMJ = (RLK, QHQ)
LTB = (XMH, STF)
VVL = (XKG, RCL)
LKZ = (DGB, RDJ)
KNS = (GCV, DHH)
GPP = (VVD, VJM)
VQQ = (DKC, VVS)
VKL = (XST, RLM)
NMG = (QRJ, LDJ)
GJF = (HTG, PSG)
VBT = (LBP, BPL)
BPD = (QMC, QJP)
KVP = (SDX, HRD)
MFB = (BJV, CBR)
PNC = (NFM, GGB)
BHJ = (NXX, RXF)
PFQ = (VTG, NFB)
RKB = (FVM, SPH)
TVB = (NBR, CCH)
HMG = (HMJ, PLF)
NKG = (QGQ, KTF)
DLD = (CLK, SPQ)
RRS = (RSM, GPP)
RJV = (HDJ, NSX)
LXH = (MFP, RHS)
CMV = (DMC, TMB)
GKB = (NBR, CCH)
VPN = (BPL, LBP)
FBF = (JHP, PDL)
CCL = (GHX, BMF)
KKP = (NPG, RPL)
MKL = (LCD, SJC)
RSM = (VVD, VJM)
SVN = (BKH, XXF)
QRM = (DMT, KFQ)
CGS = (BCB, HFB)
XQB = (DPL, VNL)
QTG = (VVS, DKC)
KRG = (CLX, PTG)
DDT = (CVT, DKH)
SFB = (KNF, CRX)
GGQ = (XKN, PVK)
PPQ = (KHF, QLN)
DMJ = (SQB, VCS)
PLJ = (GPS, NNN)
BKP = (QJK, CLJ)
FSG = (NFB, VTG)
VBF = (PLM, QVM)
QRT = (PCB, FHH)
DMT = (KDF, LMH)
DTJ = (DRV, CRJ)
CMB = (FQJ, RLT)
MKP = (MHT, JTB)
RHB = (JSG, FQL)
PVJ = (XHM, NMC)
BHP = (VLX, VLX)
DLA = (SSF, PXF)
JXB = (QVG, QVC)
VCS = (XHH, LPR)
PCS = (HRS, VTK)
HPC = (DRH, RGF)
CJP = (PSR, BPR)
GBD = (TRS, LDN)
MDJ = (KDR, KVP)
SJD = (QVG, QVC)
QCX = (PLM, QVM)
JHM = (VVL, KDZ)
TFJ = (CMV, KJG)
GLF = (RXF, NXX)
SXQ = (TCG, QMN)
JKL = (VVL, VVL)
RQS = (PVM, KCQ)
VTG = (BHJ, GLF)
VGL = (FBQ, NFG)
XTM = (LPQ, JNG)
FBQ = (LGC, CFS)
XCL = (FDF, VNF)
VXV = (SVM, HQD)
MQS = (NRD, LTB)
MDH = (FJQ, KMC)
DHT = (SPH, FVM)
JHG = (JKL, JHM)
QVN = (KLG, CDN)
DTB = (VDL, VCM)
LDN = (BRL, FLC)
MSX = (SNR, DLD)
JTM = (NNC, PDH)
RMR = (FTV, LLH)
MTH = (XJM, HKT)
FVM = (QTG, VQQ)
LSF = (DNC, VPQ)
JHS = (FNJ, JGH)
CXH = (VKF, HHL)
FHT = (QPR, MXT)
QPL = (PTG, CLX)
NJF = (VDR, JTM)
MBD = (FLP, QFJ)
NSX = (MBH, QRM)
MBM = (JDD, MFX)
HHF = (PBK, GSJ)
DPM = (FSX, VKL)
JFG = (JJP, CMB)
KHD = (NPF, BXL)
DDS = (CSQ, PHG)
QLV = (PJL, CTJ)
CLJ = (HCQ, VMT)
MRS = (DSR, DSL)
GHG = (DXX, JGD)
""",
    )
