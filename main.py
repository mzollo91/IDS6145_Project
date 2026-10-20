# %% Setting Global Constants and Deterministic Functions
# Imports
import math

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


def prob_of_detection(phi: float, alpha: float, beta: float, p_max: float) -> float:
    """
    Calculate the probability of detection using a log-logistic-type model.

    Parameters:
        phi    : Flaw size (signal input)
        alpha  : Scale parameter (flaw size at 50% detection, for beta-shape)
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
    saw_min_eff: float = SAW_MIN_EFF,
    saw_max_eff: float = SAW_MAX_EFF,
    saw_min_slag_coef: float = SAW_MIN_SLAG_COEF,
    saw_max_slag_coef: float = SAW_MAX_SLAG_COEF,
    steel_density: float = STEEL_DENSITY,
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
    if shell_id < 44:
        root_gap = 0.0625
    else:
        root_gap = 0.125

    weld_area = root_gap * (0.5 * shell_thick + 0.0625)
    total_weld_mass = weld_area * shell_length * steel_density

    min_slag = (total_weld_mass / saw_max_eff) * saw_min_slag_coef / steel_density
    max_slag = (total_weld_mass / saw_min_eff) * saw_max_slag_coef / steel_density
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
    
def coupling_slag(shellThick, outputChoice=0, SMAW_minEff=SMAW_MIN_EFF, SMAW_maxEff=SMAW_MAX_EFF,
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

    if outputChoice == 0:
        return avgTotalSlag
    elif outputChoice == -1:
        return minTotalSlag
    elif outputChoice == 1:
        return maxTotalSlag
    else:
        return avgTotalSlag
    
def station_2_slag(shell_ID, outputChoice=0, SAW_minEff=SAW_MIN_EFF, SAW_maxEff=SAW_MAX_EFF,
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

    if outputChoice == 0:
        return avgSlag
    elif outputChoice == -1:
        return minSlag
    elif outputChoice == 1:
        return maxSlag
    else:
        return avgSlag

def station_3_slag(shell_ID, shellLength, vesselType, outputChoice=0,
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

    # Create a dictionary of shell IDs and lengths to large nozzle sizes.

    # vesselCombos format is "shell ID_shell length"
    vesselCombos = ["25.25_120", "25.25_144", "29.25_120", "29.25_144", "29.25_168", "33.25_144", "33.25_168", "39_144", "39_168", "39_192", "44_144", "44_168", "44_192", "48_144",
                    "48_168", "48_192", "56_168", "56_192", "56_216", "60_168", "60_192", "60_216", "64_192", "64_216", "68_192", "68_216"]

    # Each array in nozzle array is (Hot Gas OD, Liquid Line OD, Suction or Discharge OD)
    nozzleArrays = [(2.375, 2.375, 10.75), (2.375, 2.375, 10.75), (2.375, 3.5, 10.75), (2.375, 3.5, 10.75), (2.375, 3.5, 12.75), (2.375, 3.5, 12.75),
                     (2.375, 3.5, 14), (3.5, 4.5, 14), (3.5, 4.5, 14), (3.5, 4.5, 16), (3.5, 6.625, 14), (3.5, 6.625, 16), (3.5, 6.625, 16),
                     (4.5, 6.625, 16), (4.5, 6.625, 16), (4.5, 6.625, 18), (5.563, 8.625, 16), (5.563, 8.625, 18), (5.563, 8.625, 18), (6.625, 8.625, 18),
                     (6.625, 8.625, 20), (6.625, 8.625, 20), (6.625, 10.75, 20), (6.625, 10.75, 24), (6.625, 10.75, 24), (6.625, 10.75, 24)]

    # Create dictionary for suction to discharge pairings.
    suctionNozzles = [10.75, 12.75, 14, 16, 18, 20, 24]
    dischargeNozzles = [8.625, 8.625, 10.75, 12, 14, 16, 18]

    suctionToDischargeDict = {}
    for i in range(len(suctionNozzles)):
        suctionToDischargeDict[suctionNozzles[i]] = dischargeNozzles[i]

    largeNozzleDict = {}
    for i in range(len(vesselCombos)):
        largeNozzleDict[vesselCombos[i]] = nozzleArrays[i]

    tempLengthAndIDString = str(shell_ID) + "_" + str(shellLength)

    if tempLengthAndIDString not in largeNozzleDict:
        return 9999

    largeNozzleArray = largeNozzleDict[tempLengthAndIDString]

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
    if vesselType == "Condenser":
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

    totalWeldMass = weldArea * (2 * pi * (largeNozzleArray[2] / 2)) * steelDensity
    minTotalSlag = minTotalSlag + (totalWeldMass / SMAW_maxEff) * SMAW_minSlagCoef / steelDensity
    maxTotalSlag = maxTotalSlag + (totalWeldMass / SMAW_minEff) * SMAW_maxSlagCoef / steelDensity
    avgTotalSlag = (minTotalSlag + maxTotalSlag) / 2

    tubesheetSlag = station_2_slag(shell_ID, outputChoice)

    if outputChoice == 0:
        return avgTotalSlag + tubesheetSlag
    elif outputChoice == -1:
        return minTotalSlag + tubesheetSlag
    elif outputChoice == 1:
        return maxTotalSlag + tubesheetSlag
    else:
        return avgTotalSlag + tubesheetSlag

def softmax(diameterRange, lengthRange, vesselTypeRange, tubeCountRange, heatTransferAreaRange,
            binRange, shell_ID, shellLength, vesselType, tubeCount, heatTransferArea,
            bin, beta_cap=-0.5, beta_fp=-1.5, beta_cost=-2.25):

    dataArrayUBound = -1

    for rowNum in range(len(diameterRange)):
        if binRange[rowNum] == bin and vesselTypeRange[rowNum] == vesselType:
            dataArrayUBound = dataArrayUBound + 1

    if dataArrayUBound <= 0:
        return 1

    i = 0
    footprint = shell_ID * shellLength

    dataArray = [[0.0] * (dataArrayUBound + 1) for _ in range(3)]

    for rowNum in range(len(diameterRange)):
        if binRange[rowNum] == bin and vesselTypeRange[rowNum] == vesselType:
            dataArray[0][i] = diameterRange[rowNum] * lengthRange[rowNum]      # Footprint proxy
            dataArray[1][i] = tubeCountRange[rowNum]                           # Cost proxy
            dataArray[2][i] = heatTransferAreaRange[rowNum]                    # Capacity proxy
            i = i + 1

    # Normalize the data for the utility function
    maxUtil = -1e30
    sumExps = 0

    utils = [0.0] * (len(dataArray[0]))

    for i in range(len(dataArray[0])):

        # Normalize the footprint proxy.
        tempMin = min(dataArray[0])
        tempMax = max(dataArray[0])

        footprintNorm = (dataArray[0][i] - tempMin) / (tempMax - tempMin)

        # Normalize the cost proxy.
        tempMin = min(dataArray[1])
        tempMax = max(dataArray[1])

        costNorm = (dataArray[1][i] - tempMin) / (tempMax - tempMin)

        # Normalize the capacity proxy.
        tempMin = min(dataArray[2])
        tempMax = max(dataArray[2])
        tempAvg = sum(dataArray[2]) / len(dataArray[2])

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

    footprintNorm = (footprint - tempMin) / (tempMax - tempMin)

    # Normalize the cost proxy.
    tempMin = min(dataArray[1])
    tempMax = max(dataArray[1])

    costNorm = (tubeCount - tempMin) / (tempMax - tempMin)

    # Normalize the capacity proxy.
    tempMin = min(dataArray[2])
    tempMax = max(dataArray[2])
    tempAvg = sum(dataArray[2]) / len(dataArray[2])

    capacityNorm = abs(heatTransferArea - tempAvg) / max(abs(tempMax - tempAvg), abs(tempAvg - tempMin))

    # Utility function based on Consumer Choice Theory.
    # beta values are unitless since parameters are min-max normalized. Typical scale is 0 to -3.0, with greater magnitude indicating importance.

    util = beta_cap * capacityNorm + beta_fp * footprintNorm + beta_cost * costNorm
    expNumerator = math.exp(util - maxUtil)

    for i in range(len(utils)):
        sumExps = sumExps + math.exp(utils[i] - maxUtil)

    return expNumerator / sumExps
