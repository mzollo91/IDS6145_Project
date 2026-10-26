# %% Setting Global Constants and Deterministic Functions
# Imports
import math
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

# Global Constants
STUB_LOSS = 2 / 14
BACKING_RING_EFF = 0.95
SAW_MIN_EFF = 0.99
SAW_MAX_EFF = 0.99
SAW_MIN_SLAG_COEF = 1
SAW_MAX_SLAG_COEF = 1.1
SMAW_MIN_EFF = 0.6
SMAW_MAX_EFF = 0.65
SMAW_MIN_SLAG_COEF = 1 - SMAW_MAX_EFF - STUB_LOSS
SMAW_MAX_SLAG_COEF = 1 - SMAW_MIN_EFF - STUB_LOSS
STEEL_DENSITY = 0.284          # lb / in^3
SAW_MIN_WELD_RATE = 12         # lb/hr
SAW_MAX_WELD_RATE = 25         # lb/hr
SMAW_MIN_WELD_RATE = 2         # lb/hr
SMAW_MAX_WELD_RATE = 4         # lb/hr
VESSEL_TYPES = ["Evaporator", "Condenser"]
FILLS = [0.65, 0.75, 0.85, 0.95]
TUBE_OD = 0.75
MAX_CYCLES = 20
TUBE_INSERT_SPEED = 12.0 # in/s
TUBESHEET_THICK = 1.625
N_VESSELS = 20
SCENARIOS = {"None": {1: False,  # Inspection scenarios to run through.
                      2: False,
                      34: False},
            "Station 1": {1: True,
                          2: False,
                          34: False},
            "Station 2": {1: False,
                          2: True,
                          34: False},
            "Station 3 & 4": {1: False,
                              2: False,
                              34: True},
            "Station 1 & 2": {1: True,
                              2: True,
                              34: False},
            "Station 1, 3 & 4": {1: True,
                                 2: False,
                                 34: True},
            "Station 2, 3 & 4": {1: False,
                                 2: True,
                                 34: True},
            "All": {1: True,
                    2: True,
                    34: True},
                }

def prob_of_detection(phi: float, alpha: float, beta: float, p_max: float) -> float:
    """
    Calculate the probability of detection using a log-logistic-type model.

    Parameters:
        phi    : Flaw size (signal input)
        alpha  : Scale parameter (flaw size at p_max/2, for beta-shape)
        beta   : Shape parameter
        p_max  : Maximum probability of detection

    Returns:
        Probability of detection = p_max * (phi^beta / (alpha^beta + phi^beta))
    """
    return p_max * ((phi ** beta) / ((alpha ** beta) + (phi ** beta)))

def get_shell_thick(shell_id: float) -> float:
    """
    Return the shell thickness based on the shell inside diameter.

    Thresholds (inclusive upper bounds):
        shell_id <= 25.25  -> 0.25
        shell_id <= 33.25  -> 0.375
        shell_id <= 48     -> 0.5
        shell_id <= 68     -> 0.625
        otherwise          -> 0.75
    """
    if shell_id <= 25.25:
        return 0.25
    elif shell_id <= 33.25:
        return 0.375
    elif shell_id <= 48:
        return 0.5
    elif shell_id <= 68:
        return 0.625
    else:
        return 0.75

def station_1_slag(
    
    shell_id: float,
    shell_length: float,
    output_choice: int = 0,
    SAW_minEff: float = SAW_MIN_EFF,
    SAW_maxEff: float = SAW_MAX_EFF,
    SAW_minSlagCoef: float = SAW_MIN_SLAG_COEF,
    SAW_maxSlagCoef: float = SAW_MAX_SLAG_COEF,
    steelDensity: float = STEEL_DENSITY,
) -> float:
    """
    Calculates the slag quantity for Station 1 (longitudinal welding and
    small couplings).

    output_choice:
        -1 -> minimum slag
         0 -> average slag (default)
         1 -> maximum slag
        any other value -> average slag
    """
    shell_thick = get_shell_thick(shell_id)

    # Root gap depends on shell diameter
    if shell_thick < 0.5:
        root_gap = 0.0625
    else:
        root_gap = 0.125

    weld_area = root_gap * (0.5 * shell_thick + 0.0625)
    total_weld_mass = weld_area * shell_length * steelDensity

    min_slag = (total_weld_mass / SAW_maxEff) * SAW_minSlagCoef / steelDensity
    max_slag = (total_weld_mass / SAW_minEff) * SAW_maxSlagCoef / steelDensity
    avg_slag = (min_slag + max_slag) / 2

    coupling = coupling_slag(shell_thick, output_choice)

    if output_choice == 0:
        return avg_slag + coupling
    elif output_choice == -1:
        return min_slag + coupling
    elif output_choice == 1:
        return max_slag + coupling
    else:
        return avg_slag + coupling
    
def coupling_slag(shellThick, output_choice=0, SMAW_minEff=SMAW_MIN_EFF, SMAW_maxEff=SMAW_MAX_EFF,
                   SMAW_minSlagCoef=SMAW_MIN_SLAG_COEF, SMAW_maxSlagCoef=SMAW_MAX_SLAG_COEF,
                   steelDensity=STEEL_DENSITY):

    couplingDict = {}
    pi = math.pi

    minTotalSlag = 0
    maxTotalSlag = 0
    avgTotalSlag = 0

    # tempArray arguments are as follows: tempArray(Coupling OD, Weld Type, Fillet Size, Root Gap, Root Face)

    # Pressure Sensor
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (0.75, tempWeldString, 0.1875, 0.0625, 0.0625)
    couplingDict["Pressure Sensor"] = tempArray

    # Relief Valve
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Double Fillet"

    tempArray = (2.25, tempWeldString, 0.3125, 0.0625, 0.0625)
    couplingDict["Relief Valve"] = tempArray

    # Sight Glass
    if shellThick <= 0.375:
        tempWeldString = "Single Bevel Groove w/ Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (2.875, tempWeldString, 0.375, 0.0625, 0.0625)
    couplingDict["Sight Glass"] = tempArray

    # Drain
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (1.75, tempWeldString, 0.3125, 0.0625, 0.0625)
    couplingDict["Drain"] = tempArray

    # Transfer
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (1.25, tempWeldString, 0.25, 0.0625, 0.0625)
    couplingDict["Transfer"] = tempArray

    # Motor/Oil Cooling
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (1, tempWeldString, 0.25, 0.0625, 0.0625)
    couplingDict["Motor/Oil Cooling"] = tempArray

    for item in couplingDict.values():

        if item[1] == "Single Bevel Groove w/ Fillet":
            weldArea = item[3] * item[4]
        else:
            weldArea = item[2] ** 2 / 2

        totalWeldMass = weldArea * (2 * pi * item[0] / 2) * steelDensity

        minTotalSlag = minTotalSlag + (totalWeldMass / SMAW_maxEff) * SMAW_minSlagCoef / steelDensity
        maxTotalSlag = maxTotalSlag + (totalWeldMass / SMAW_minEff) * SMAW_maxSlagCoef / steelDensity
        avgTotalSlag = (minTotalSlag + maxTotalSlag) / 2

    if output_choice == 0:
        return avgTotalSlag
    elif output_choice == -1:
        return minTotalSlag
    elif output_choice == 1:
        return maxTotalSlag
    else:
        return avgTotalSlag
    
def station_2_slag(shell_ID, output_choice=0, SAW_minEff=SAW_MIN_EFF, SAW_maxEff=SAW_MAX_EFF,
                    SAW_minSlagCoef=SAW_MIN_SLAG_COEF, SAW_maxSlagCoef=SAW_MAX_SLAG_COEF,
                    steelDensity=STEEL_DENSITY, backingRingEff=BACKING_RING_EFF):

    # Station 2 is one tubesheet weld and tube insertion. Tube debris is a separate function.
    
    pi = math.pi

    shellThick = get_shell_thick(shell_ID)

    if shellThick < 0.5:
        rootGap = 0.0625
    else:
        rootGap = 0.125

    if shell_ID < 48:
        rootFace = 0.09375
    else:
        rootFace = 0.15625

    weldArea = rootGap * rootFace
    totalWeldMass = weldArea * (2 * pi * (shell_ID + 2 * shellThick) / 2) * steelDensity
    minSlag = (totalWeldMass / SAW_maxEff) * SAW_minSlagCoef / steelDensity * (1 - backingRingEff)
    maxSlag = (totalWeldMass / SAW_minEff) * SAW_maxSlagCoef / steelDensity * (1 - backingRingEff)
    avgSlag = (minSlag + maxSlag) / 2

    if output_choice == 0:
        return avgSlag
    elif output_choice == -1:
        return minSlag
    elif output_choice == 1:
        return maxSlag
    else:
        return avgSlag

def station_3_slag(shell_ID, shell_length, vessel_type, largeNozzleDict, suctionToDischargeDict, output_choice=0,
                    SMAW_minEff=SMAW_MIN_EFF, SMAW_maxEff=SMAW_MAX_EFF,
                    SMAW_minSlagCoef=SMAW_MIN_SLAG_COEF, SMAW_maxSlagCoef=SMAW_MAX_SLAG_COEF,
                    steelDensity=STEEL_DENSITY):

    # Station 3 is all large nozzle welding and installation of the second tubesheet.

    shellThick = get_shell_thick(shell_ID)

    # rootGap and rootFace are consistent between all nozzles.
    rootGap = 0.0625
    rootFace = 0.0625

    pi = math.pi

    minTotalSlag = 0
    maxTotalSlag = 0
    avgTotalSlag = 0

    tempLengthAndIDTuple = (shell_ID, shell_length)

    if tempLengthAndIDTuple not in largeNozzleDict:
        raise ValueError(f"Nozzle configuration not found for shell ID {shell_ID} and length {shell_length}.")

    largeNozzleArray = largeNozzleDict[tempLengthAndIDTuple]

    # Hot gas nozzle.
    if largeNozzleArray[0] < 3.5:
        tempWeldString = "Single Bevel Groove w/ Fillet"
    else:
        tempWeldString = "Double Fillet"

    if largeNozzleArray[0] < 4.5:
        filletSize = 0.25
    else:
        filletSize = 0.375

    if tempWeldString == "Single Bevel Groove w/ Fillet":
        weldArea = rootGap * rootFace
    else:
        weldArea = filletSize ** 2 / 2

    totalWeldMass = weldArea * (2 * pi * (largeNozzleArray[0] / 2)) * steelDensity
    minTotalSlag = minTotalSlag + (totalWeldMass / SMAW_maxEff) * SMAW_minSlagCoef / steelDensity
    maxTotalSlag = maxTotalSlag + (totalWeldMass / SMAW_minEff) * SMAW_maxSlagCoef / steelDensity
    avgTotalSlag = (minTotalSlag + maxTotalSlag) / 2

    # Liquid line nozzle.
    tempWeldString = "Single Bevel Groove w/ Fillet"
    if largeNozzleArray[1] < 6.625:
        filletSize = 0.25
    else:
        filletSize = 0.375

    if tempWeldString == "Single Bevel Groove w/ Fillet":
        weldArea = rootGap * rootFace
    else:
        weldArea = filletSize ** 2 / 2

    totalWeldMass = weldArea * (2 * pi * (largeNozzleArray[1] / 2)) * steelDensity
    minTotalSlag = minTotalSlag + (totalWeldMass / SMAW_maxEff) * SMAW_minSlagCoef / steelDensity
    maxTotalSlag = maxTotalSlag + (totalWeldMass / SMAW_minEff) * SMAW_maxSlagCoef / steelDensity
    avgTotalSlag = (minTotalSlag + maxTotalSlag) / 2

    # Suction or discharge nozzle.
    if vessel_type == "Condenser":
        governingNozzleOD = suctionToDischargeDict[largeNozzleArray[2]]
    else:
        governingNozzleOD = largeNozzleArray[2]

    if governingNozzleOD == 14 or governingNozzleOD == 16:
        tempWeldString = "Double Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    if governingNozzleOD < 14:
        filletSize = 0.25
    else:
        filletSize = 0.375

    if tempWeldString == "Single Bevel Groove w/ Fillet":
        weldArea = rootGap * rootFace
    else:
        weldArea = filletSize ** 2 / 2

    totalWeldMass = weldArea * (2 * pi * (governingNozzleOD / 2)) * steelDensity
    minTotalSlag = minTotalSlag + (totalWeldMass / SMAW_maxEff) * SMAW_minSlagCoef / steelDensity
    maxTotalSlag = maxTotalSlag + (totalWeldMass / SMAW_minEff) * SMAW_maxSlagCoef / steelDensity
    avgTotalSlag = (minTotalSlag + maxTotalSlag) / 2

    tubesheetSlag = station_2_slag(shell_ID, output_choice)

    if output_choice == 0:
        return avgTotalSlag + tubesheetSlag
    elif output_choice == -1:
        return minTotalSlag + tubesheetSlag
    elif output_choice == 1:
        return maxTotalSlag + tubesheetSlag
    else:
        return avgTotalSlag + tubesheetSlag

def softmax(diameterList, lengthList, vessel_typeList, tubeCountList, heatTransferAreaList,
            binList, shell_ID, shell_length, vessel_type, tubeCount, heatTransferArea,
            areaBin, beta_cap=-0.5, beta_fp=-1.5, beta_cost=-2.25):

    dataArrayUBound = -1

    for rowNum in range(len(diameterList)):
        if binList[rowNum] == areaBin and vessel_typeList[rowNum] == vessel_type:
            dataArrayUBound = dataArrayUBound + 1

    if dataArrayUBound <= 0:
        return 1

    i = 0
    footprint = shell_ID * shell_length

    dataArray = [[0.0] * (dataArrayUBound + 1) for _ in range(3)]

    for rowNum in range(len(diameterList)):
        if binList[rowNum] == areaBin and vessel_typeList[rowNum] == vessel_type:
            dataArray[0][i] = diameterList[rowNum] * lengthList[rowNum]      # Footprint proxy
            dataArray[1][i] = tubeCountList[rowNum]                           # Cost proxy
            dataArray[2][i] = heatTransferAreaList[rowNum]                    # Capacity proxy
            i = i + 1

    # Normalize the data for the utility function
    maxUtil = -1e30
    sumExps = 0

    utils = [0.0] * (len(dataArray[0]))

    for i in range(len(dataArray[0])):

        # Normalize the footprint proxy.
        tempMin = min(dataArray[0])
        tempMax = max(dataArray[0])

        if tempMax - tempMin == 0:
            footprintNorm = 0
        else:
            footprintNorm = (dataArray[0][i] - tempMin) / (tempMax - tempMin)

        # Normalize the cost proxy.
        tempMin = min(dataArray[1])
        tempMax = max(dataArray[1])

        if tempMax - tempMin == 0:
            costNorm = 0
        else:
            costNorm = (dataArray[1][i] - tempMin) / (tempMax - tempMin)

        # Normalize the capacity proxy.
        tempMin = min(dataArray[2])
        tempMax = max(dataArray[2])
        tempAvg = sum(dataArray[2]) / len(dataArray[2])

        if tempMax - tempAvg == 0 and tempAvg - tempMin == 0:
            capacityNorm = 0
        else:
            capacityNorm = abs(dataArray[2][i] - tempAvg) / max(abs(tempMax - tempAvg), abs(tempAvg - tempMin))

        # Utility function based on Consumer Choice Theory.
        # beta values are unitless since parameters are min-max normalized. Typical scale is 0 to -3.0, with greater magnitude indicating importance.

        utils[i] = beta_cap * capacityNorm + beta_fp * footprintNorm + beta_cost * costNorm
        if utils[i] > maxUtil:
            maxUtil = utils[i]

    # For the SoftMax formula, calculate the numerator based on the inputs from the specific row.

    # Normalize the footprint proxy.
    tempMin = min(dataArray[0])
    tempMax = max(dataArray[0])

    if tempMax - tempMin == 0:
        footprintNorm = 0
    else:
        footprintNorm = (footprint - tempMin) / (tempMax - tempMin)

    # Normalize the cost proxy.
    tempMin = min(dataArray[1])
    tempMax = max(dataArray[1])

    if tempMax - tempMin == 0:
        costNorm = 0
    else:
        costNorm = (tubeCount - tempMin) / (tempMax - tempMin)

    # Normalize the capacity proxy.
    tempMin = min(dataArray[2])
    tempMax = max(dataArray[2])
    tempAvg = sum(dataArray[2]) / len(dataArray[2])

    if tempMax - tempAvg == 0 and tempAvg - tempMin == 0:
        capacityNorm = 0
    else:
        capacityNorm = abs(heatTransferArea - tempAvg) / max(abs(tempMax - tempAvg), abs(tempAvg - tempMin))

    # Utility function based on Consumer Choice Theory.
    # beta values are unitless since parameters are min-max normalized. Typical scale is 0 to -3.0, with greater magnitude indicating importance.

    util = beta_cap * capacityNorm + beta_fp * footprintNorm + beta_cost * costNorm
    expNumerator = math.exp(util - maxUtil)

    for i in range(len(utils)):
        sumExps = sumExps + math.exp(utils[i] - maxUtil)

    return expNumerator / sumExps

def station_1_weld_time(shell_ID, shell_length, output_choice=0,
                         SAW_minWeldRate=SAW_MIN_WELD_RATE, SAW_maxWeldRate=SAW_MAX_WELD_RATE,
                         steelDensity=STEEL_DENSITY):
  
    shellThick = get_shell_thick(shell_ID)

    if shellThick < 0.5:
        rootGap = 0.0625
    else:
        rootGap = 0.125

    pi = math.pi

    # Roller time
    rollerSpeed = 0.1  # in/s
    shellCircum = pi * (shell_ID + 2 * shellThick)
    thicknessMultiplier = shellThick / 0.25
    rollerTime = shellCircum / (rollerSpeed * 3600) * thicknessMultiplier

    weldArea = rootGap * (0.5 * shellThick + 0.0625) * 2
    totalWeldMass = weldArea * shell_length * steelDensity
    minTime = totalWeldMass / SAW_maxWeldRate + rollerTime
    maxTime = totalWeldMass / SAW_minWeldRate + rollerTime
    avgTime = (minTime + maxTime) / 2

    couplingWeldTime = coupling_weld_time(shellThick, output_choice)

    if output_choice == 0:
        return avgTime + couplingWeldTime
    elif output_choice == -1:
        return minTime + couplingWeldTime
    elif output_choice == 1:
        return maxTime + couplingWeldTime
    else:
        return avgTime + couplingWeldTime

def coupling_weld_time(shellThick, output_choice=0, SMAW_minWeldRate=SMAW_MIN_WELD_RATE,
                        SMAW_maxWeldRate=SMAW_MAX_WELD_RATE, steelDensity=STEEL_DENSITY):

    pi = math.pi
    alpha = 45 * (pi / 180)  # radians

    minTotalTime = 0
    maxTotalTime = 0
    avgTotalTime = 0

    couplingDict = {}

    # tempArray arguments are as follows: tempArray(Coupling OD, Weld Type, Fillet Size, Root Gap, Root Face)

    # Pressure Sensor
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (0.75, tempWeldString, 0.1875, 0.0625, 0.0625)
    couplingDict["Pressure Sensor"] = tempArray

    # Relief Valve
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Double Fillet"

    tempArray = (2.25, tempWeldString, 0.3125, 0.0625, 0.0625)
    couplingDict["Relief Valve"] = tempArray

    # Sight Glass
    if shellThick <= 0.375:
        tempWeldString = "Single Bevel Groove w/ Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (2.875, tempWeldString, 0.375, 0.0625, 0.0625)
    couplingDict["Sight Glass"] = tempArray

    # Drain
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (1.75, tempWeldString, 0.3125, 0.0625, 0.0625)
    couplingDict["Drain"] = tempArray

    # Transfer
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (1.25, tempWeldString, 0.25, 0.0625, 0.0625)
    couplingDict["Transfer"] = tempArray

    # Motor/Oil Cooling
    if shellThick <= 0.375:
        tempWeldString = "Single Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    tempArray = (1, tempWeldString, 0.25, 0.0625, 0.0625)
    couplingDict["Motor/Oil Cooling"] = tempArray

    for item in couplingDict.values():

        if item[1] == "Single Bevel Groove w/ Fillet":
            weldArea = (item[3] * shellThick) + (shellThick - item[4]) ** 2 * math.tan(alpha / 2) + (item[2] ** 2) / 2
        else:
            weldArea = item[2] ** 2 / 2

        totalWeldMass = weldArea * (2 * pi * item[0] / 2) * steelDensity

        minTotalTime = minTotalTime + (totalWeldMass / SMAW_maxWeldRate)
        maxTotalTime = maxTotalTime + (totalWeldMass / SMAW_minWeldRate)
        avgTotalTime = (minTotalTime + maxTotalTime) / 2

    if output_choice == 0:
        return avgTotalTime
    elif output_choice == -1:
        return minTotalTime
    elif output_choice == 1:
        return maxTotalTime
    else:
        return avgTotalTime

def station_2_weld_time(shell_ID, output_choice=0, SAW_minWeldRate=SAW_MIN_WELD_RATE, SAW_maxWeldRate=SAW_MAX_WELD_RATE,
                   steelDensity=STEEL_DENSITY):

    shellThick = get_shell_thick(shell_ID)

    if shellThick < 0.5:
        rootGap = 0.0625
    else:
        rootGap = 0.125

    if shell_ID < 48:
        rootFace = 0.09375
    else:
        rootFace = 0.15625

    pi = math.pi
    alpha = 45 * (pi / 180)  # radians

    weldArea = (rootGap * shellThick) + (shellThick - rootFace) ** 2 * math.tan(alpha / 2) + (shellThick ** 2) / 2
    totalWeldMass = weldArea * (2 * pi * (shell_ID + 2 * shellThick) / 2) * steelDensity
    minTime = totalWeldMass / SAW_maxWeldRate
    maxTime = totalWeldMass / SAW_minWeldRate
    avgTime = (minTime + maxTime) / 2

    if output_choice == 0:
        return avgTime
    elif output_choice == -1:
        return minTime
    elif output_choice == 1:
        return maxTime
    else:
        return avgTime

def station_3_weld_time(shell_ID, shell_length, vessel_type, largeNozzleDict, suctionToDischargeDict, output_choice=0,
                         SMAW_minWeldRate=SMAW_MIN_WELD_RATE, SMAW_maxWeldRate=SMAW_MAX_WELD_RATE,
                         steelDensity=STEEL_DENSITY):

    # Station 3 is all large nozzle welding and installation of the second tubesheet.

    shellThick = get_shell_thick(shell_ID)

    # rootGap and rootFace are consistent between all nozzles.
    rootGap = 0.0625
    rootFace = 0.0625

    pi = math.pi
    alpha = 45 * (pi / 180)  # radians

    minTotalTime = 0
    maxTotalTime = 0
    avgTotalTime = 0

    tempLengthAndIDTuple = (shell_ID, shell_length)

    if tempLengthAndIDTuple not in largeNozzleDict:
        raise ValueError(f"Nozzle configuration not found for shell ID {shell_ID} and length {shell_length}.")

    largeNozzleArray = largeNozzleDict[tempLengthAndIDTuple]

    # Hot gas nozzle.
    if largeNozzleArray[0] < 3.5:
        tempWeldString = "Single Bevel Groove w/ Fillet"
    else:
        tempWeldString = "Double Fillet"

    if largeNozzleArray[0] < 4.5:
        filletSize = 0.25
    else:
        filletSize = 0.375

    if tempWeldString == "Single Bevel Groove w/ Fillet":
        weldArea = (rootGap * shellThick) + (shellThick - rootFace) ** 2 * math.tan(alpha / 2) + (filletSize ** 2) / 2
    else:
        weldArea = filletSize ** 2 / 2 * 2

    totalWeldMass = weldArea * (2 * pi * (largeNozzleArray[0] / 2)) * steelDensity
    minTotalTime = minTotalTime + totalWeldMass / SMAW_maxWeldRate
    maxTotalTime = maxTotalTime + totalWeldMass / SMAW_minWeldRate
    avgTotalTime = (minTotalTime + maxTotalTime) / 2

    # Liquid line nozzle.
    tempWeldString = "Single Bevel Groove w/ Fillet"
    if largeNozzleArray[1] < 6.625:
        filletSize = 0.25
    else:
        filletSize = 0.375

    if tempWeldString == "Single Bevel Groove w/ Fillet":
        weldArea = (rootGap * shellThick) + (shellThick - rootFace) ** 2 * math.tan(alpha / 2) + (filletSize ** 2) / 2
    else:
        weldArea = filletSize ** 2 / 2 * 2

    totalWeldMass = weldArea * (2 * pi * (largeNozzleArray[1] / 2)) * steelDensity
    minTotalTime = minTotalTime + totalWeldMass / SMAW_maxWeldRate
    maxTotalTime = maxTotalTime + totalWeldMass / SMAW_minWeldRate
    avgTotalTime = (minTotalTime + maxTotalTime) / 2

    # Suction or discharge nozzle.
    if vessel_type == "Condenser":
        governingNozzleOD = suctionToDischargeDict[largeNozzleArray[2]]
    else:
        governingNozzleOD = largeNozzleArray[2]

    if governingNozzleOD == 14 or governingNozzleOD == 16:
        tempWeldString = "Double Fillet"
    else:
        tempWeldString = "Single Bevel Groove w/ Fillet"

    if governingNozzleOD < 14:
        filletSize = 0.25
    else:
        filletSize = 0.375

    if tempWeldString == "Single Bevel Groove w/ Fillet":
        weldArea = (rootGap * shellThick) + (shellThick - rootFace) ** 2 * math.tan(alpha / 2) + (filletSize ** 2) / 2
    else:
        weldArea = filletSize ** 2 / 2 * 2

    totalWeldMass = weldArea * (2 * pi * (governingNozzleOD / 2)) * steelDensity
    minTotalTime = minTotalTime + totalWeldMass / SMAW_maxWeldRate
    maxTotalTime = maxTotalTime + totalWeldMass / SMAW_minWeldRate
    avgTotalTime = (minTotalTime + maxTotalTime) / 2

    tubesheetTime = station_2_weld_time(shell_ID, output_choice)

    if output_choice == 0:
        return avgTotalTime + tubesheetTime
    elif output_choice == -1:
        return minTotalTime + tubesheetTime
    elif output_choice == 1:
        return maxTotalTime + tubesheetTime
    else:
        return avgTotalTime + tubesheetTime

@lru_cache(maxsize=None)
def get_copper_debris(shell_length, tube_od=TUBE_OD, skippless_length = 168.0,
                     mu_k_finned=0.38, mu_k_skip=0.18, massPerFoot=0.31,
                     supportThick=0.375, tubesheetThick=TUBESHEET_THICK,
                     wearCoefficient=0.00001, indentationHardness=70):

    pi = math.pi
    tubeEndSkip = 2  # All tubes have a 2" skip at the end for tubesheet expansion.
    hardnessPsi = indentationHardness * 1422.3  # Approximate conversion for Vicker's Hardness to PSI for HX copper tubes.
    
    skippless = True if shell_length <= skippless_length else False

    if skippless:
        supportSpacing = 21
    else:
        supportSpacing = 42

    if skippless:
        skipLength = 0
    else:
        skipLength = 2

    if shell_length % supportSpacing > 0:
        numOfSupports = shell_length // supportSpacing + 1
        outerSupportSpacing = (shell_length % supportSpacing - tubesheetThick * 2 - supportThick) / 2
    else:
        numOfSupports = shell_length // supportSpacing - 1
        outerSupportSpacing = (shell_length - (supportSpacing * numOfSupports) - tubesheetThick * 2 - supportThick) / 2

    # Outline the skip positioning if a skipped design.
    if skippless:
        skipRanges = [[0, 0]]
    else:
        skipRanges = [None] * (numOfSupports + 1)
        for i in range(0, numOfSupports):
            r0 = (outerSupportSpacing + tubesheetThick + supportSpacing * i) - skipLength / 2
            r1 = (outerSupportSpacing + tubesheetThick + supportSpacing * i) + skipLength / 2
            skipRanges[i] = [r0, r1]
        skipRanges[numOfSupports] = [shell_length + 2, shell_length + 2]

    tubesheetContactArea = tube_od * pi * tubesheetThick
    volumetricWear = 0
    rangeTracker = 0
    totalInsertionDepth = tubesheetThick
    totalSupportContactArea = 0

    # The initial push into the tubesheet to the first support must be kept separate from the for loop given its slightly different nature.
    # Copper tube is assumed to be inserted the depth of the tubesheet.
    stepSize = 0.0625

    overhangLength = 0
    while True:

        if not skippless:
            if totalInsertionDepth > skipRanges[rangeTracker][1]:
                rangeTracker = rangeTracker + 1

        N_weight_ts = totalInsertionDepth / 12 * massPerFoot

        # All tube holes have a 45° chamfer, so the sag force is calculated as the vertical component of the normal force.
        N_sag = ((overhangLength ** 2 / 12 * massPerFoot) / 2) / (tubesheetThick / 12) * math.cos(45 * pi / 180)

        if (overhangLength <= tubeEndSkip - tubesheetThick) or \
           (totalInsertionDepth > skipRanges[rangeTracker][0] and totalInsertionDepth < skipRanges[rangeTracker][1]):
            pushForce = (N_weight_ts + N_sag) * mu_k_skip
        else:
            pushForce = (N_weight_ts + N_sag) * mu_k_finned

        volumetricWear = volumetricWear + wearCoefficient * (pushForce * totalInsertionDepth) / hardnessPsi

        if overhangLength == outerSupportSpacing:
            break

        totalInsertionDepth = totalInsertionDepth + stepSize

        overhangLength = overhangLength + stepSize
        if overhangLength > outerSupportSpacing:
            break

    for i in range(1, numOfSupports + 1):

        # The tube has not been inserted into the support at this point, a small code block is needed
        # to account for inserting through the thickness of the support.

        mu_k_support = mu_k_skip

        supportInsertionDepth = stepSize
        while supportInsertionDepth <= supportThick:

            totalInsertionDepth = totalInsertionDepth + stepSize

            if not skippless:
                if totalInsertionDepth > skipRanges[rangeTracker][1]:
                    rangeTracker = rangeTracker + 1

            # Calculate the forces from the tubesupport. The skipped end of the tube will be first
            totalSupportContactArea = totalSupportContactArea + stepSize * tube_od * pi

            # Calculate weight at the tubesheet, assumed an even distribution amongst supports.
            N_weight_ts = totalInsertionDepth / 12 * massPerFoot / (i + 1)

            # Since the weight is assumed equal at all supports, the weight at each support is equal to the N_weight_ts * numOfSupports
            N_weight_support = N_weight_ts * i

            # Determine coefficient of friction based on skip position at the tubesheet.
            if totalInsertionDepth > skipRanges[rangeTracker][0] and totalInsertionDepth < skipRanges[rangeTracker][1]:
                mu_k_tubesheet = mu_k_skip
            else:
                mu_k_tubesheet = mu_k_finned

            pushForce = N_weight_ts * mu_k_tubesheet + N_weight_support * mu_k_support

            volumetricWear = volumetricWear + wearCoefficient * (pushForce * totalInsertionDepth) / hardnessPsi

            supportInsertionDepth = supportInsertionDepth + stepSize

        if i == numOfSupports:
            governingSpacing = outerSupportSpacing
        else:
            governingSpacing = supportSpacing - supportThick

        overhangLength = stepSize
        while overhangLength <= governingSpacing:

            totalInsertionDepth = totalInsertionDepth + stepSize

            if not skippless:
                if totalInsertionDepth > skipRanges[rangeTracker][1]:
                    rangeTracker = rangeTracker + 1

            # Calculate weight at the tubesheet, assumed an even distribution amongst supports.
            N_weight_ts = totalInsertionDepth / 12 * massPerFoot / (i + 1)

            # Since the weight is assumed equal at all supports, the weight at each support is equal to the N_weight_ts * numOfSupports
            N_weight_support = N_weight_ts * i

            # Determine coefficient of friction based on skip position at the tubesheet.
            if totalInsertionDepth > skipRanges[rangeTracker][0] and totalInsertionDepth < skipRanges[rangeTracker][1]:
                mu_k_tubesheet = mu_k_skip
            else:
                mu_k_tubesheet = mu_k_finned

            # Calculate the force from the moment of the overhanging tube on the support. The same 45° chamfer is here as well.
            N_sag = ((overhangLength ** 2 / 12 * massPerFoot) / 2) / (supportThick / 12) * math.cos(45 * pi / 180)

            if (overhangLength <= tubeEndSkip - supportThick) or \
               (overhangLength > skipRanges[0][0] and overhangLength < skipRanges[0][1]):
                mu_k_support = mu_k_skip
            else:
                mu_k_support = mu_k_finned

            pushForce = N_weight_ts * mu_k_tubesheet + (N_weight_support + N_sag) * mu_k_support

            volumetricWear = volumetricWear + wearCoefficient * (pushForce * totalInsertionDepth) / hardnessPsi

            overhangLength = overhangLength + stepSize

    # The tube is inserted into the last tubesheet.

    mu_k_tubesheet = mu_k_skip
    skipOffset = tubesheetThick
    tubesheetInsertionDepth = 0

    tubesheetInsertionDepth = stepSize
    while tubesheetInsertionDepth <= tubesheetThick:

        totalInsertionDepth = totalInsertionDepth + stepSize
        if not skippless:
            skipOffset = skipOffset - stepSize
            if skipOffset > skipLength / 2:
                mu_k_support = mu_k_finned
            else:
                mu_k_support = mu_k_skip

        if not skippless:
            if totalInsertionDepth > skipRanges[rangeTracker][1]:
                rangeTracker = rangeTracker + 1

        # Calculate weight at the tubesheet, assumed an even distribution amongst supports.
        N_weight_ts = (totalInsertionDepth / 12 * massPerFoot / (numOfSupports + 2)) * 2

        # Since the weight is assumed equal at all supports, the weight at each support is equal to the N_weight_ts * numOfSupports
        N_weight_support = N_weight_ts * numOfSupports

        pushForce = N_weight_ts * mu_k_tubesheet + N_weight_support * mu_k_support

        volumetricWear = volumetricWear + wearCoefficient * (pushForce * totalInsertionDepth) / hardnessPsi

        tubesheetInsertionDepth = tubesheetInsertionDepth + stepSize

    return volumetricWear

def available_tube_area(shell_id, vessel_type):

    if vessel_type == "Evaporator":

        top_r = 0.3 * shell_id / 2
        bottom_r = 0.25 * shell_id / 2
        side_r = 0.25 * shell_id / 2
        part1 = (top_r + bottom_r) * side_r * 2

        theta = math.acos((shell_id / 2) * 0.5 / (shell_id / 2)) * 2
        part2 = 0.5 * (shell_id / 2) ** 2 * (theta - math.sin(theta))

        return part1 + part2

    else:

        top_theta = math.acos((shell_id / 2) * 0.6 / (shell_id / 2)) * 2
        bottom_theta = math.acos((shell_id / 2) * 0.4 / (shell_id / 2)) * 2
        top_segment = 0.5 * (shell_id / 2) ** 2 * (top_theta - math.sin(top_theta))
        bottom_segment = 0.5 * (shell_id / 2) ** 2 * (bottom_theta - math.sin(bottom_theta))

        result = math.pi * (shell_id / 2) ** 2 - top_segment - bottom_segment

        return result * (1 - shell_id / 100)

def num_of_tubes_used(available_tube_area, percentage_used, tube_od=TUBE_OD) -> int:
    single_tube_area = (tube_od/2)**2 * math.pi
    return int(np.floor(available_tube_area * percentage_used / single_tube_area))

def heat_transfer_area(shell_length, num_of_tubes, tube_od=TUBE_OD) -> float:
    single_tube_circumference = math.pi * tube_od
    return num_of_tubes * single_tube_circumference * shell_length

def sturges_bins(n):
    """Sturges' formula: k = log2(n) + 1."""
    k_raw = np.log2(n) + 1
    k     = int(np.ceil(k_raw)) # Adjusting from np.floor to np.ceil to round up.
    return k, k_raw

def phi(debris, shell_id, shell_length):
    """Calculate the phi value based on debris at a given station and the bottom 10% of the shell volume."""
    return debris / (math.pi * (shell_id / 2) ** 2 * shell_length * 0.1)

def station_2_alpha_multiplier(shell_id, available_tube_area, percentage_hx_area_used):
    b = (available_tube_area * percentage_hx_area_used) / (math.pi * (shell_id / 2) ** 2)
    return 1 / (1-b)

def station_3_alpha_multiplier(shell_id, available_tube_area, percentage_hx_area_used):
    b = (available_tube_area * percentage_hx_area_used) / (math.pi * (shell_id / 2) ** 2)
    return 1 / (1-b) * 2

# Create a dictionary of shell IDs and lengths to large nozzle sizes.

# vesselCombos format is "shell ID, shell length"
vesselCombos = [[25.25, 120], [25.25, 144], [29.25, 120], [29.25, 144], [29.25, 168], [33.25, 144], [33.25, 168], [39, 144], [39, 168], [39, 192], [44, 144], [44, 168], [44, 192], [48, 144],
                [48, 168], [48, 192], [56, 168], [56, 192], [56, 216], [60, 168], [60, 192], [60, 216], [64, 192], [64, 216], [68, 192], [68, 216]]

# Each array in nozzle array is (Hot Gas OD, Liquid Line OD, Suction or Discharge OD)
nozzleArrays = [(2.375, 2.375, 10.75), (2.375, 2.375, 10.75), (2.375, 3.5, 10.75), (2.375, 3.5, 10.75), (2.375, 3.5, 12.75), (2.375, 3.5, 12.75),
                    (2.375, 3.5, 14), (3.5, 4.5, 14), (3.5, 4.5, 14), (3.5, 4.5, 16), (3.5, 6.625, 14), (3.5, 6.625, 16), (3.5, 6.625, 16),
                    (4.5, 6.625, 16), (4.5, 6.625, 16), (4.5, 6.625, 18), (5.563, 8.625, 16), (5.563, 8.625, 18), (5.563, 8.625, 18), (6.625, 8.625, 18),
                    (6.625, 8.625, 20), (6.625, 8.625, 20), (6.625, 10.75, 20), (6.625, 10.75, 24), (6.625, 10.75, 24), (6.625, 10.75, 24)]

# Create dictionary for suction to discharge pairings.
suctionNozzles = [10.75, 12.75, 14, 16, 18, 20, 24]
dischargeNozzles = [8.625, 8.625, 10.75, 12, 14, 16, 18]

suctionToDischargeDict = dict((zip(suctionNozzles, dischargeNozzles)))

vesselCombos = [tuple(x) for x in vesselCombos]
largeNozzleDict = dict(zip(vesselCombos, nozzleArrays))

# Parameters to sweep.
# "role" indicates whether a parameter is swept, sampled, or fixed.
# "swept" is a fixed but uncertain value.
# "sampled" is a per-event random variable. Will use the triangular distribution to sample values.
# "fixed" is a parameter with no variability or stochasticity.

PARAMS = {
    "s1_cleaning_factor": dict(min=0.9, max=1.0, mode=0.98, role="swept"), # Station 1 cleaning factor, dimensionless.
    "s2_cleaning_factor": dict(min=0.4, max=1.0, mode=0.55, role="swept"), # Station 2 cleaning factor, dimensionless.
    "s3_cleaning_factor": dict(min=0.0, max=0.0, mode=0.0, role="fixed"), # Station 3 cleaning factor, dimensionless.
    "s4_cleaning_factor": dict(min=0.98, max=1.0, mode=0.995, role="swept"), # Station 4 cleaning factor, dimensionless.
    "s1_rework_time": dict(min=0.16667, max=0.75, mode=0.25, role="sampled", distribution="triangular"), # Station 1 rework time, hours.
    "s2_rework_time": dict(min=0.5, max=1.5, mode=0.75, role="sampled", distribution="triangular"), # Station 2 rework time, hours.
    "s3_rework_time": dict(min=0.0, max=0.0, mode=0.0, role="fixed"), # Station 3 rework time, hours.
    "s4_rework_time": dict(min=12.0, max=36.0, mode=16.0, role="sampled", distribution="triangular"), # Station 4 rework time, hours.
    "workmanship_quality": dict(min=0.95, max=1.15, mode=1.0, role="sampled", distribution="triangular"), # Workmanship quality factor, dimensionless.
    "debris_detection_rate": dict(min=0.7, max=0.85, mode=0.8, role="swept"), # Debris detection rate (percentage chance of finding debris), dimensionless.
    "debris_ratio_alpha_median": dict(min=0.00001, max=0.0001, mode=0.00005, role="swept"), # Median debris ratio that can be detected by an inspector when the detection reaches about 1/2 of pMax (alpha in POD function).
    "debris_detection_beta": dict(min=0.9, max=1.1, mode=1.0, role="swept"), # Rate at which detection increases around the median debris ratio (beta in POD function).
    "vessel_arrival_rate": dict(min=0.567648402, max=0.73390411, mode=0.652639269, role="swept"), # Average arrival rate of vessels to the station (vessel/hour), shared across 5 individual stations.
    "s4_processing_time": dict(min=2.0, max=5.0, mode=2.5, role="sampled", distribution="triangular"), # Manufacturing process time at station 4 (hours): waterbox & closure install, pressure test. Pressure tests can take up to 4 hours if leaks are found.
    "s1_breakdown_rate": dict(min=2.06954E-05, max=4.13907E-05, mode=3.44923E-05, role="swept"), # Rate of failure from the roller at station 1 (breakdown/hour). Failures are rare, maintenance is more likely, ranging every 5 to 10 years.
    "s1_repair_time": dict(min=16.0, max=160.0, mode=48.0, role="sampled", distribution="triangular"), # How long it takes to repair the roller at station 1 (hours). Rollers can be down anywhere from one day to two weeks depending on maintenance.
    "s1_inspection_time": dict(min=0.166666667, max=0.333333333, mode=0.25, role="sampled", distribution="triangular"), # How long it takes to inspect the vessel at station 1 (hours): quick visual inspection of the empty shell.
    "s2_inspection_time": dict(min=1.0, max=2.0, mode=1.5, role="sampled", distribution="triangular"), # How long it takes to inspect the vessel at station 2 (hours): boroscope inspection through couplings and one end of the tubesheet.
    "s34_inspection_time": dict(min=3.0, max=5.0, mode=3.75, role="sampled", distribution="triangular"), # How long it takes to inspect the vessel at station 3 or 4 (hours): boroscope inspection through couplings and one end of the tubesheet.
    "evap_share": dict(min=0.5, max=1.0, mode=0.8, role="swept"), # Percentage share of evaporators to condensers coming down the line.
}
# %% Building the dataframe of all possible vessel combinations and their associated parameters.
skeleton_data = product(vesselCombos, VESSEL_TYPES, FILLS)
df = pd.DataFrame({'catalog_data': skeleton_data})

df[["Tuple", "Vessel Type", "% Tube Area Used"]] = pd.DataFrame(df["catalog_data"].tolist(), index=df.index)
df[["Shell Diameter", "Shell Length"]] = pd.DataFrame(df["Tuple"].tolist(), index=df.index)
df = df.drop(columns=["catalog_data", "Tuple"], inplace=False)
df = df[["Shell Diameter", "Shell Length", "Vessel Type", "% Tube Area Used"]]

df["Available Tube Area"] = df.apply(lambda row: available_tube_area(row["Shell Diameter"], row["Vessel Type"]), axis=1)
df["Tube Count"] = df.apply(lambda row: num_of_tubes_used(row["Available Tube Area"], row["% Tube Area Used"]), axis=1)
df["Copper Debris"] = df.apply(lambda row: get_copper_debris(row["Shell Length"]) * row["Tube Count"], axis=1)
df["Station 1 Slag"] = df.apply(lambda row: station_1_slag(row["Shell Diameter"], row["Shell Length"], output_choice=0), axis=1)
df["Station 2 Slag"] = df.apply(lambda row: station_2_slag(row["Shell Diameter"], output_choice=0), axis=1)
df["Station 3 Slag"] = df.apply(lambda row: station_3_slag(row["Shell Diameter"], row["Shell Length"], row["Vessel Type"], largeNozzleDict, suctionToDischargeDict, output_choice=0), axis=1)
df["Phi - Station 1"] = df.apply(lambda row: phi(row["Station 1 Slag"], row["Shell Diameter"], row["Shell Length"]), axis=1)
df["Phi - Station 2"] = df.apply(lambda row: phi(row["Station 2 Slag"] + row["Copper Debris"], row["Shell Diameter"], row["Shell Length"]), axis=1)
df["Phi - Station 3"] = df.apply(lambda row: phi(row["Station 3 Slag"], row["Shell Diameter"], row["Shell Length"]), axis=1)
df["Heat Transfer Area"] = df.apply(lambda row: heat_transfer_area(row["Shell Length"], row["Tube Count"]), axis=1)
df["Station 2 Alpha Multiplier"] = df.apply(lambda row: station_2_alpha_multiplier(row["Shell Diameter"], row["Available Tube Area"], row["% Tube Area Used"]), axis=1)
df["Station 3 Alpha Multiplier"] = df.apply(lambda row: station_3_alpha_multiplier(row["Shell Diameter"], row["Available Tube Area"], row["% Tube Area Used"]), axis=1)

df_evap = df[df["Vessel Type"] == "Evaporator"]
df_cond = df[df["Vessel Type"] == "Condenser"]

evap_bins, _ = sturges_bins(len(df_evap))
cond_bins, _ = sturges_bins(len(df_cond))

min_evap, max_evap = df_evap['Heat Transfer Area'].min(), df_evap['Heat Transfer Area'].max()
min_cond, max_cond = df_cond['Heat Transfer Area'].min(), df_cond['Heat Transfer Area'].max()

evap_bin_edges = np.linspace(min_evap, max_evap, evap_bins + 1)
cond_bin_edges = np.linspace(min_cond, max_cond, cond_bins + 1)

df["Bin"] = None
evap_mask = df["Vessel Type"] == "Evaporator"
cond_mask = df["Vessel Type"] == "Condenser"

df.loc[evap_mask, "Bin"] = pd.cut(df.loc[evap_mask, "Heat Transfer Area"], bins=evap_bin_edges, labels=False, include_lowest=True)
df.loc[cond_mask, "Bin"] = pd.cut(df.loc[cond_mask, "Heat Transfer Area"], bins=cond_bin_edges, labels=False, include_lowest=True)

df.loc[evap_mask, "Bin Probability"] = df["Bin"].map(df.loc[evap_mask, "Bin"].value_counts() / len(df_evap))
df.loc[cond_mask, "Bin Probability"] = df["Bin"].map(df.loc[cond_mask, "Bin"].value_counts() / len(df_cond))

df["Softmax"] = df.apply(lambda row: softmax(df["Shell Diameter"].tolist(), df["Shell Length"].tolist(), df["Vessel Type"].tolist(), df["Tube Count"].tolist(), df["Heat Transfer Area"].tolist(),
                  df["Bin"].tolist(), row["Shell Diameter"], row["Shell Length"], row["Vessel Type"], row["Tube Count"], row["Heat Transfer Area"], row["Bin"]), axis=1)

df["Probability Weight"] = df.apply(lambda row: row["Softmax"] * row["Bin Probability"], axis=1)

# df.to_csv("skeleton_data.csv", index=False)
# %% Build the engine for the debris, inspection and rework.

def tri(rng, p): # the PARAMS dict that is min/max/mode. NumPy expects left, mode, right for the triangular distribution.
    if not (p["min"] <= p["mode"] <= p["max"]):
        raise ValueError(f"Invalid triangular bounds: Must satisfy min ({p["min"]}) <= mode ({p["mode"]}) <= max ({p["max"]})")
    if p["min"] == p["max"]:
        return p["min"]
    return rng.triangular(p["min"], p["mode"], p["max"])

def station_1(v, w1, base, sampled, inspect_here, rng_inspect, state):
    d, l = v["Shell Diameter"], v["Shell Length"]
    s1_slag = v["Station 1 Slag"]
    state["debris"] += s1_slag * w1
    state["debris"] *= 1 - base["s1_cleaning_factor"]
    state["service_hours"] += station_1_weld_time(shell_ID=d, shell_length=l, output_choice=0) * w1

    if inspect_here:
        alpha = base["debris_ratio_alpha_median"]
        beta = base["debris_detection_beta"]
        p_max = base["debris_detection_rate"]

        for _ in range(MAX_CYCLES):
            state["hours"] += tri(rng_inspect, sampled["s1_inspection_time"])
            pod = prob_of_detection(phi=phi(debris=state["debris"],shell_id=d,shell_length=l),alpha=alpha, beta=beta, p_max=p_max)
            if rng_inspect.random() >= pod:
                break # If missed or no debris is in the shell, exit the loop.
            state["debris"] *= 1 - base["s1_cleaning_factor"]
            state["hours"] += tri(rng_inspect, sampled["s1_rework_time"])
            state["cycles"] += 1
    return state

def station_2(v, w2, base, sampled, inspect_here, rng_inspect, state, tube_insert_speed=TUBE_INSERT_SPEED):
    d, l, t = v["Shell Diameter"], v["Shell Length"], v["Tube Count"]
    s2_slag, cop_debris = v["Station 2 Slag"], v["Copper Debris"]
    s2_alpha_mult = v["Station 2 Alpha Multiplier"]
    state["debris"] += s2_slag * w2
    state["debris"] += cop_debris * w2
    state["debris"] *= 1 - base["s2_cleaning_factor"]
    state["service_hours"] += station_2_weld_time(shell_ID=d, output_choice=0) * w2
    state["service_hours"] +=  (l / tube_insert_speed / 3600) * t * w2 # based on insertion speed of a single tube.

    if inspect_here:
        alpha = base["debris_ratio_alpha_median"] * s2_alpha_mult
        beta = base["debris_detection_beta"]
        p_max = base["debris_detection_rate"]

        for _ in range(MAX_CYCLES):
            state["hours"] += tri(rng_inspect, sampled["s2_inspection_time"])
            pod = prob_of_detection(phi=phi(debris=state["debris"],shell_id=d,shell_length=l),alpha=alpha, beta=beta, p_max=p_max)
            if rng_inspect.random() >= pod:
                break # If missed or no debris is in the shell, exit the loop.
            state["debris"] *= 1 - base["s2_cleaning_factor"]
            state["hours"] += tri(rng_inspect, sampled["s2_rework_time"])
            state["cycles"] += 1
    return state

def station_3_and_4(v, w3, w4, pt4, largeNozzleDict, suctionToDischargeDict, base, sampled, inspect_here, rng_inspect, state, tube_insert_speed=TUBE_INSERT_SPEED, tubesheet_thick=TUBESHEET_THICK):
    d, l, t, vt = v["Shell Diameter"], v["Shell Length"], v["Tube Count"], v["Vessel Type"]
    s3_slag = v["Station 3 Slag"]
    s3_alpha_mult = v["Station 3 Alpha Multiplier"]
    state["debris"] += s3_slag * w3
    state["service_hours"] += station_3_weld_time(shell_ID=d, shell_length=l, vessel_type=vt, largeNozzleDict=largeNozzleDict, suctionToDischargeDict=suctionToDischargeDict) * w3
    state["service_hours"] +=  (tubesheet_thick / tube_insert_speed / 3600) * t * w3 # Inserting the rest of the tube into the second tubesheet.

    if inspect_here:
        alpha = base["debris_ratio_alpha_median"] * s3_alpha_mult
        beta = base["debris_detection_beta"]
        p_max = base["debris_detection_rate"]

        for _ in range(MAX_CYCLES):
            state["hours"] += tri(rng_inspect, sampled["s34_inspection_time"])
            pod = prob_of_detection(phi=phi(debris=state["debris"],shell_id=d,shell_length=l),alpha=alpha, beta=beta, p_max=p_max)
            if rng_inspect.random() >= pod:
                break # If missed or no debris is in the shell, exit the loop.
            state["debris"] *= 1 - base["s4_cleaning_factor"]
            state["hours"] += tri(rng_inspect, sampled["s4_rework_time"])
            state["cycles"] += 1

    state["service_hours"] += pt4 * w4 # Closure and waterbox install, then pressure test lumped together.

    return state

base = {k: v["mode"] for k, v in PARAMS.items() if v["role"] == "swept"}
sampled = {k: v for k, v in PARAMS.items() if v["role"] == "sampled" or v["role"] == "fixed"}

seeds = np.random.SeedSequence(12345).spawn(20) # one stream per replication.
catalog = df.to_dict("records") # A faster alternative to df.loc inside loops.
share = base["evap_share"]
w = df["Probability Weight"] * np.where(df["Vessel Type"] == "Evaporator", share, 1 - share)
assert np.isclose(w.sum(), 1.0), f"Probability weights do not sum to 1.0, they sum to {w.sum()}."
results = []

for rep, s in enumerate(seeds):
    # Calling 'spawn' will return different children each time, for both seed sequences it should be called in the seed loop for conistency in replications.
    vessel_ss, inspect_ss = s.spawn(2) # separate streams by purpose.
    rng_vessel = np.random.default_rng(vessel_ss)
    vessel_streams = inspect_ss.spawn(N_VESSELS)

    v = catalog[rng_vessel.choice(len(catalog), p=w.to_numpy())] # Randomly select a vessel based on the probability weights.

    # Worksmanship quality factors for each station, sampled once per replication.
    # Processing time for station four is not scenario dependent. This should be drawn at the same time as the vessel for consistency.
    batch = [[catalog[rng_vessel.choice(len(catalog), p=w.to_numpy())],[tri(rng_vessel, PARAMS["workmanship_quality"]) for _ in range(4)],
              tri(rng_vessel, sampled["s4_processing_time"])] for _ in range(N_VESSELS)]

    for name, flags in SCENARIOS.items():
        print(f"Now running inspection scenario '{name}'.")
        for (v, w_st, s4_t), child in zip(batch, vessel_streams):
            rng_inspect = np.random.default_rng(inspect_ss)    
            state = dict(debris=0.0, hours=0.0, cycles=0, service_hours=0.0)
            
            station_1(v, w_st[0], base, sampled, flags[1], rng_inspect, state)
            station_2(v, w_st[1], base, sampled, flags[2], rng_inspect, state)
            station_3_and_4(v, w_st[2], w_st[3], s4_t, largeNozzleDict, suctionToDischargeDict, base, sampled, flags[34], rng_inspect, state)
            #print(v["Shell Diameter"], v["Shell Length"], state)
            results.append(dict(rep=rep, scenario=name, escaped=state["debris"], hours=state["hours"], cycles=state["cycles"]))

for result in results:
    for key, value in result.items():
        print(f"{key}: {value}")
# %%
