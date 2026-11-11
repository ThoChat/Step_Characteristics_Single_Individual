"""Reference copies of the previous analysis functions (DeCIMo-analysis).

Copied from ``AnalysisBasics.py`` and ``AllQty_Dataframe.py`` with only these
changes: explicit numpy imports instead of ``from pylab import *``, debug
plotting and printing removed, and two blocks of ``CreateAllQtyDF`` wrapped in
functions: the first step (:func:`FirstStepBlock`) and the minimal time to
boundary with the empirical perturbation angle (:func:`PerturbationAngleBlock`).
They are the ground truth of ``tests/`` and of
``notebooks/02_Analysis_Test.ipynb``: the new implementation must return the
same values.
"""

import matplotlib.path as mplPath
import numpy as np
from numpy import argmax, argmin, array, pad, sign, sqrt, zeros, shape
from scipy.signal import savgol_filter
from shapely.geometry import LineString

dt = 1 / 200


def TemporalDeriv(X, dt):
    if len(shape(X)) > 1:
        DerivX = zeros(X.shape)
        DerivX[:, 0] = (4 * X[:, 1] - 3 * X[:, 0] - X[:, 2]) / (2 * dt)
        DerivX[:, -1] = (-4 * X[:, -2] + 3 * X[:, -1] + X[:, -3]) / (2 * dt)
        for i in range(1, len(X[0]) - 1):
            DerivX[:, i] = (X[:, i + 1] - X[:, i - 1]) / (2 * dt)
    else:
        DerivX = zeros(len(X))
        DerivX[0] = (4 * X[1] - 3 * X[0] - X[2]) / (2 * dt)
        DerivX[-1] = (-4 * X[-2] + 3 * X[-1] + X[-3]) / (2 * dt)
        for i in range(1, len(X) - 1):
            DerivX[i] = (X[i + 1] - X[i - 1]) / (2 * dt)
    return DerivX


def FindPeaks(Curve, Threshold, ThresholdInflexion, Epsilon=0, RealSartIndex=0):
    if len(Curve) < 3:
        return []
    if max(Curve) < Threshold:
        return []
    MaxIndex = argmax(Curve)
    if MaxIndex > 0:
        i = MaxIndex
        while i > 0 and (Curve[i] - Curve[i - 1] > Epsilon or Curve[i] >= ThresholdInflexion):
            i -= 1
        BegInflexionIndex = i
    else:
        BegInflexionIndex = 0
    if MaxIndex < len(Curve) - 1:
        i = MaxIndex
        while i < len(Curve) - 1 and (Curve[i] - Curve[i + 1] > Epsilon or Curve[i] >= ThresholdInflexion):
            i += 1
        EndInflexionIndex = i
    else:
        EndInflexionIndex = len(Curve) - 1
    LeftPeaks = FindPeaks(Curve[:BegInflexionIndex], Threshold, ThresholdInflexion,
                          Epsilon=Epsilon, RealSartIndex=RealSartIndex)
    RightPeaks = FindPeaks(Curve[EndInflexionIndex:], Threshold, ThresholdInflexion,
                           Epsilon=Epsilon, RealSartIndex=RealSartIndex + EndInflexionIndex)
    peak = [[RealSartIndex + BegInflexionIndex, RealSartIndex + EndInflexionIndex]]
    return LeftPeaks + peak + RightPeaks


def norm(vector):
    return sqrt(vector[0] ** 2 + vector[1] ** 2)


def vect(a, b):
    return array([b[0] - a[0], b[1] - a[1]])


def dist_point(a, b):
    x, y = vect(a, b)
    return sqrt(x**2 + y**2)


def StepDetector(RTOE_Data, LTOE_Data, dt, DistThreshold=2e-2, AbductionThreshold=3e-2,
                 SuccessfulAbductionThreshold=5e-2, LocalMaxThreshold=0.3, TransVelThreshold=0.1,
                 Epsilon=0.01, FramesIgnored=50, exception=False):
    ListOfStep = [[], []]
    for FootIndex, TOE in enumerate([RTOE_Data, LTOE_Data]):
        FTOE = pad(TOE[:, FramesIgnored:-FramesIgnored], ((0, 0), (50, 0)), mode="edge")
        DistToe = array([dist_point(FTOE[:2, 0], toe_i[:2]) for toe_i in FTOE.T])
        if max(DistToe) < DistThreshold:
            continue
        if (abs(max(DistToe) - DistToe[-1]) > AbductionThreshold) and not (
            exception or DistToe[-1] > SuccessfulAbductionThreshold
        ):
            ListOfStep[FootIndex] = ListOfStep[FootIndex] + ["Abduction"]
            continue
        VRTOE = array([TemporalDeriv(ax, dt) for ax in FTOE[:2]])
        Vtrans = [norm([Vrt[0], Vrt[1]]) for Vrt in VRTOE.T]
        ValueTransVelThreshold = TransVelThreshold * max(Vtrans)
        ValueLocalMaxThreshold = LocalMaxThreshold * max(Vtrans)
        NormalisedEpsilon = Epsilon * max(Vtrans)
        AllPeaks = FindPeaks(Vtrans, ValueLocalMaxThreshold, ValueTransVelThreshold, NormalisedEpsilon)
        for peak in AllPeaks:
            if dist_point(FTOE[:2, peak[0]], FTOE[:2, peak[1]]) > DistThreshold:
                ListOfStep[FootIndex] = ListOfStep[FootIndex] + [peak]
    if any([x == "Abduction" for x in ListOfStep[0] + ListOfStep[1]]):
        if len(ListOfStep[0]) == 0 or len(ListOfStep[1]) == 0:
            return ["lAbd"], [[1, -1]]
        else:
            return ["lAbd+Ul"], [[1, -1]]
    n1 = len(ListOfStep[0])
    n2 = len(ListOfStep[1])
    if n1 + n2 == 0:
        return ["X"], [[1, -1]]
    AllStepsSorted = [0] * (n1 + n2)
    SteppingFootList = ["R"] * (n1 + n2)
    i1 = i2 = i = 0
    while i1 < n1 and i2 < n2:
        if ListOfStep[0][i1][0] < ListOfStep[1][i2][0]:
            AllStepsSorted[i] = ListOfStep[0][i1]
            i1 += 1
        else:
            AllStepsSorted[i] = ListOfStep[1][i2]
            SteppingFootList[i] = "L"
            i2 += 1
        i += 1
    while i1 < n1:
        AllStepsSorted[i] = ListOfStep[0][i1]
        i1 += 1
        i += 1
    while i2 < n2:
        AllStepsSorted[i] = ListOfStep[1][i2]
        SteppingFootList[i] = "L"
        i2 += 1
        i += 1
    return SteppingFootList, AllStepsSorted


def ComputePerturbationDuration(ForceData, Treshold=0.25):
    StartIndex = 0
    EndIndex = len(ForceData) - 1
    FilteredForce = savgol_filter(ForceData, 55, 5)
    FDeriv = TemporalDeriv(FilteredForce, dt) / max(abs(TemporalDeriv(FilteredForce, dt)))
    i = argmax(FDeriv)
    while StartIndex == 0 and i > 0:
        if FDeriv[i] <= 0 and ForceData[i] / max(ForceData) < Treshold:
            StartIndex = i
        i -= 1
    i = argmin(FDeriv)
    while EndIndex == len(ForceData) - 1 and i < len(ForceData) - 1:
        if FDeriv[i] >= 0 and ForceData[i] / max(ForceData) < Treshold:
            EndIndex = i
        i += 1
    if max(ForceData - ForceData[StartIndex]) < 5:
        StartIndex, EndIndex = 0, 1
    return StartIndex, EndIndex


def ComputeImpulse(ForceData, ImpulseStart, ImpulseEnd, Tstep):
    ForceData = ForceData - ForceData[-1]
    Impulse = 0
    for i in range(ImpulseStart, ImpulseEnd):
        Impulse += (ForceData[i] + ForceData[i + 1]) * 0.5
    return Impulse * Tstep


def FirstStepBlock(RTOE, RTARI, RHEE, RTAR, LTOE, LTARI, LHEE, LTAR, COM, exception=False):
    """First-step block of ``CreateAllQtyDF`` (trials with StepIndex == 2).

    ``exception`` is added: ``CreateAllQtyDF`` always re-ran ``StepDetector``
    in default mode, while ``GenerateStepDataframe`` used ``exception=True``
    for 90 deg pushes. Both gave the same steps on the previous data.

    Returns
    -------
    dict
        SteppingFoot, FirstStepBegIndex, FirstStepEndIndex, CrossoverIndex,
        UnifiedClassifStepName, StepVector, FirstStepDist, FirstStepDuration,
        StepLength_xVcomRef, StepLength_yVcomRef.
    """
    Nbframes = len(COM[0])
    VCOM = array([TemporalDeriv(x, dt) for x in COM])
    SteppingFootList, AllStepsIndexSorted = StepDetector(RTOE, LTOE, dt, exception=exception)
    SteppingFoot = SteppingFootList[0]
    FirstStepBegIndex = AllStepsIndexSorted[0][0]
    FirstStepEndIndex = AllStepsIndexSorted[0][1]
    FirstEscapeCoMSpeed = norm(VCOM[:2, FirstStepBegIndex])

    CrossoverIndex = max(
        -sign(vect(LTOE[:2, FirstStepBegIndex], RTOE[:2, FirstStepBegIndex]).dot(
            vect(LTOE[:2, FirstStepEndIndex], RTOE[:2, FirstStepEndIndex]))),
        -sign(vect(LHEE[:2, FirstStepBegIndex], RHEE[:2, FirstStepBegIndex]).dot(
            vect(LHEE[:2, FirstStepEndIndex], RHEE[:2, FirstStepEndIndex]))),
    )
    if SteppingFoot == "R":
        NSFTOE, NSFTARI, NSFTAR, NSFHEE = LTOE, LTARI, LTAR, LHEE
        SFTOE, SFTARI, SFTAR, SFHEE = RTOE, RTARI, RTAR, RHEE
    else:
        NSFTOE, NSFTARI, NSFTAR, NSFHEE = RTOE, RTARI, RTAR, RHEE
        SFTOE, SFTARI, SFTAR, SFHEE = LTOE, LTARI, LTAR, LHEE

    b = FirstStepBegIndex
    NSF_BoSX = [NSFTOE[0, b], NSFTARI[0, b], NSFHEE[0, b], NSFTAR[0, b], NSFTOE[0, b]]
    NSF_BoSY = [NSFTOE[1, b], NSFTARI[1, b], NSFHEE[1, b], NSFTAR[1, b], NSFTOE[1, b]]
    NSFB_path = mplPath.Path(np.array([[NSF_BoSX[i], NSF_BoSY[i]] for i in range(4)]))

    UnifiedClassifStepName = " "
    if CrossoverIndex == 1:
        UnifiedClassifStepName = "UCS"
    else:
        TOESsegment = LineString([SFTOE[:2, b], NSFTOE[:2, b]])
        for t_idx in range(Nbframes - 1):
            if TOESsegment.intersects(LineString([COM[:2, t_idx], COM[:2, t_idx + 1]])):
                UnifiedClassifStepName = "FS"
                break
        if UnifiedClassifStepName == " ":
            SF_BoSX = [SFTOE[0, b], SFTARI[0, b], SFHEE[0, b], SFTAR[0, b], SFTOE[0, b]]
            SF_BoSY = [SFTOE[1, b], SFTARI[1, b], SFHEE[1, b], SFTAR[1, b], SFTOE[1, b]]
            SFB_path = mplPath.Path(np.array([[SF_BoSX[i], SF_BoSY[i]] for i in range(4)]))
            SCS_flag = 1
            for com_t in COM.T[b:]:
                if NSFB_path.contains_point(com_t[:2]):
                    UnifiedClassifStepName = "UMS"
                    SCS_flag = 0
                    break
                if SFB_path.contains_point(com_t[:2]):
                    UnifiedClassifStepName = "LSS"
                    SCS_flag = 0
                    break
            if SCS_flag:
                UnifiedClassifStepName = "SCS"

    Vect_X_axis_VcomRef = VCOM[:2, b] / FirstEscapeCoMSpeed
    Vect_Y_axis_VcomRef = array([Vect_X_axis_VcomRef[1], -Vect_X_axis_VcomRef[0]])
    StepVector = vect(SFTOE[:, b], SFTOE[:, FirstStepEndIndex])
    return dict(
        SteppingFoot=SteppingFoot,
        FirstStepBegIndex=FirstStepBegIndex,
        FirstStepEndIndex=FirstStepEndIndex,
        CrossoverIndex=CrossoverIndex,
        UnifiedClassifStepName=UnifiedClassifStepName,
        StepVector=StepVector,
        FirstStepDist=norm(StepVector),
        FirstStepDuration=(FirstStepEndIndex - b) * dt,
        StepLength_xVcomRef=StepVector.dot(Vect_X_axis_VcomRef),
        StepLength_yVcomRef=StepVector.dot(Vect_Y_axis_VcomRef),
    )


# ---------------------------------------------------------------------------
# Time to boundary and empirical perturbation angle (AnalysisBasics.py)
# ---------------------------------------------------------------------------
def orientation(p, q, r):
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0


def onSegment(p, q, r):
    if (
        (q[0] <= max(p[0], r[0]))
        and (q[0] >= min(p[0], r[0]))
        and (q[1] <= max(p[1], r[1]))
        and (q[1] >= min(p[1], r[1]))
    ):
        return True
    return False


def doIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if (o1 != o2) and (o3 != o4):
        return True
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True
    return False


def crossing_point(A1, B1, A2, B2):
    if (A1[0] - B1[0]) != 0 and (A2[0] - B2[0]) != 0:
        a1 = (A1[1] - B1[1]) / (A1[0] - B1[0])
        a2 = (A2[1] - B2[1]) / (A2[0] - B2[0])
        b1 = -1
        b2 = -1
        c1 = A1[1] - a1 * A1[0]
        c2 = A2[1] - a2 * A2[0]
        try:
            return array(
                [
                    (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1),
                    (c1 * a2 - a1 * c2) / (a1 * b2 - a2 * b1),
                ]
            )
        except Exception:
            return None
    elif (A1[0] - B1[0]) == 0 and (A2[0] - B2[0]) != 0:
        a2 = (A2[1] - B2[1]) / (A2[0] - B2[0])
        c2 = A2[1] - a2 * A2[0]
        return array([A1[0], a2 * A1[0] + c2])
    elif (A2[0] - B2[0]) == 0 and (A1[0] - B1[0]) != 0:
        a1 = (A1[1] - B1[1]) / (A1[0] - B1[0])
        c1 = A1[1] - a1 * A1[0]
        return array([A2[0], a1 * A2[0] + c1])
    elif (A2[0] - B2[0]) == 0 and (A1[0] - B1[0]) == 0:
        return None
    else:
        return -1


def closest_point_seg(a, b, c):
    u = array(a) - array(c)
    v = array(b) - array(a)
    t = -v.dot(u) / v.dot(v)
    if t >= 0 and t <= 1:
        return (1 - t) * array(a) + t * array(b)
    elif v.dot(v) + 2 * (v.dot(u)) + u.dot(u) > u.dot(u):
        return a
    else:
        return b


def BoSSegmentClosestCrossingPoint(XBoS_t, YBoS_t, COM_t):
    TwoDCoM_t = COM_t[:2]
    NbSeg = len(XBoS_t) - 1
    MinBoSIndex = argmin(
        [
            dist_point(
                closest_point_seg(
                    array([XBoS_t[i], YBoS_t[i]]),
                    array([XBoS_t[i + 1], YBoS_t[i + 1]]),
                    TwoDCoM_t,
                ),
                TwoDCoM_t,
            )
            for i in range(NbSeg)
        ]
    )
    return closest_point_seg(
        array([XBoS_t[MinBoSIndex], YBoS_t[MinBoSIndex]]),
        array([XBoS_t[MinBoSIndex + 1], YBoS_t[MinBoSIndex + 1]]),
        TwoDCoM_t,
    )


def BoSSegmentCrossingPoint(XBoS_t, YBoS_t, COM_t, VCOM_t, Method="Both", debug=True):
    TwoDCoM_t = COM_t[:2]
    TwoDVCoM_t = VCOM_t[:2]
    TwoDFarVCoMProjection_t = TwoDCoM_t + TwoDVCoM_t / norm(TwoDVCoM_t)
    NbSeg = len(XBoS_t) - 1

    if Method == "VelBased" or Method == "Both":
        for i in range(NbSeg):
            if doIntersect(
                array([XBoS_t[i], YBoS_t[i]]),
                array([XBoS_t[i + 1], YBoS_t[i + 1]]),
                TwoDCoM_t,
                TwoDFarVCoMProjection_t,
            ):
                return crossing_point(
                    array([XBoS_t[i], YBoS_t[i]]),
                    array([XBoS_t[i + 1], YBoS_t[i + 1]]),
                    TwoDCoM_t,
                    TwoDFarVCoMProjection_t,
                )
    if Method == "VelBased":
        return None
    if Method == "CloserPoint" or Method == "Both":
        TwoDNegFarVCoMProjection_t = TwoDCoM_t - TwoDVCoM_t / norm(TwoDVCoM_t)
        CurrentCrossingPoint = array([1e5, 1e5])
        for i in range(NbSeg):
            if doIntersect(
                array([XBoS_t[i], YBoS_t[i]]),
                array([XBoS_t[i + 1], YBoS_t[i + 1]]),
                TwoDCoM_t,
                TwoDNegFarVCoMProjection_t,
            ):
                C = crossing_point(
                    array([XBoS_t[i], YBoS_t[i]]),
                    array([XBoS_t[i + 1], YBoS_t[i + 1]]),
                    TwoDCoM_t,
                    TwoDFarVCoMProjection_t,
                )
                if dist_point(C, TwoDCoM_t) < dist_point(CurrentCrossingPoint, TwoDCoM_t):
                    CurrentCrossingPoint = C
        if debug and norm(CurrentCrossingPoint) > 1e4:
            CurrentCrossingPoint = BoSSegmentClosestCrossingPoint(XBoS_t, YBoS_t, COM_t)
        return CurrentCrossingPoint


def ChangeOfBasis(Vect3D, PassingMatrix):
    return np.array(
        [
            sum(PassingMatrix[0] * Vect3D),
            sum(PassingMatrix[1] * Vect3D),
            sum(PassingMatrix[2] * Vect3D),
        ]
    )


def PerturbationAngleBlock(RTOE, LTOE, LTAR, LHEE, RHEE, RTAR, COM, Force, FirstStepBegIndex=None):
    """Min Ttb and empirical perturbation angle blocks of ``CreateAllQtyDF``.

    ``FirstStepBegIndex`` is the first step t_begin for step trials, None for
    the other trials (the window then ends 1 s after the end of the push, as
    in the previous code).

    Returns
    -------
    dict
        MinTtb, MinTtbIndex, PerturbationAngle, MaxForce.
    """
    Nbframes = len(COM[0])
    VCOM = array([TemporalDeriv(x, dt) for x in COM])
    S, E = ComputePerturbationDuration(Force)

    BoS3D = array(
        [
            [RTOE[:, i], LTOE[:, i], LTAR[:, i], LHEE[:, i], RHEE[:, i], RTAR[:, i], RTOE[:, i]]
            for i in range(Nbframes)
        ]
    )
    XBoS = [BoS3D[i, :, 0] for i in range(Nbframes)]
    YBoS = [BoS3D[i, :, 1] for i in range(Nbframes)]

    if FirstStepBegIndex is None:
        FirstStepBegIndex = min(E + 200, Nbframes - 1)

    Nbframes_computation = len(COM[0, S:FirstStepBegIndex])
    ArrTtb = zeros(Nbframes_computation)
    for i in range(S, S + Nbframes_computation):
        C = BoSSegmentCrossingPoint(XBoS[i], YBoS[i], COM[:, i], VCOM[:, i])
        Speed_i = norm(VCOM[:2, i])
        Dtb_i = vect(COM[:2, i], C).dot(VCOM[:2, i] / Speed_i)
        ArrTtb[i - S] = Dtb_i / Speed_i
    MinTtb = min(ArrTtb)
    MinTtbIndex = S + argmin(ArrTtb)

    cosT = np.cos(np.pi / 2)
    sinT = np.sin(np.pi / 2)
    CoBmatrix = array([[cosT, sinT, 0], [-sinT, cosT, 0], [0, 0, 1]])
    VCOM_NR = array([ChangeOfBasis(x, CoBmatrix) for x in VCOM.T]).T
    PerturbationAngle = sign(VCOM_NR[1, MinTtbIndex]) * (
        np.arccos(VCOM_NR[:2, MinTtbIndex][0] / norm(VCOM_NR[:2, MinTtbIndex])) * 180 / np.pi
    )
    return dict(
        MinTtb=MinTtb,
        MinTtbIndex=MinTtbIndex,
        PerturbationAngle=PerturbationAngle,
        MaxForce=max(Force - Force[-1]),
    )
