"""
Helper module for reading and writing relion star files.
"""
from pathlib import PurePath
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union, overload
from typing_extensions import Literal
import numpy as n
from numpy.core.records import fromrecords

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore

from .util import topen

# Available star file fields and their types. Fields marked as type ``object``
# should always be interepreted as strings.
RLN_DTYPES: Dict[str, Type[object]] = dict(
    rlnComment=object,
    # Area
    rlnAreaId=int,
    rlnAreaName=object,
    # Body
    rlnBodyMaskName=object,
    rlnBodyKeepFixed=int,
    rlnBodyReferenceName=object,
    rlnBodyRotateDirectionX=float,
    rlnBodyRotateDirectionY=float,
    rlnBodyRotateDirectionZ=float,
    rlnBodyRotateRelativeTo=int,
    rlnBodySigmaAngles=float,
    rlnBodySigmaOffset=float,
    rlnBodySigmaOffsetAngst=float,
    rlnBodySigmaRot=float,
    rlnBodySigmaTilt=float,
    rlnBodySigmaPsi=float,
    rlnBodyStarFile=object,
    # CTF
    rlnCtfAstigmatism=float,
    rlnCtfBfactor=float,
    rlnCtfMaxResolution=float,
    rlnCtfValidationScore=float,
    rlnCtfScalefactor=float,
    rlnVoltage=float,
    rlnDefocusU=float,
    rlnDefocusV=float,
    rlnDefocusAngle=float,
    rlnSphericalAberration=float,
    rlnChromaticAberration=float,
    rlnDetectorPixelSize=float,
    rlnCtfPowerSpectrum=object,
    rlnEnergyLoss=float,
    rlnCtfFigureOfMerit=float,
    rlnCtfImage=object,
    rlnLensStability=float,
    rlnMagnification=float,
    rlnPhaseShift=float,
    rlnConvergenceCone=float,
    rlnLongitudinalDisplacement=float,
    rlnTransversalDisplacement=float,
    rlnAmplitudeContrast=float,
    rlnCtfValue=float,
    # Image
    rlnImageName=object,
    rlnImageOriginalName=object,
    rlnReconstructImageName=object,
    rlnImageId=int,
    rlnEnabled=bool,
    rlnDataType=int,
    rlnImageDimensionality=int,
    rlnBeamTiltX=float,
    rlnBeamTiltY=float,
    rlnMtfFileName=object,
    rlnOpticsGroup=int,
    rlnOpticsGroupName=object,
    # rlnOddZernike='f8 VECTOR',  # TODO: Determine correct interpretation
    # rlnEvenZernike='f8 VECTOR',  # TODO: Determine correct interpretation
    rlnImagePixelSize=float,
    rlnMagMat00=float,
    rlnMagMat01=float,
    rlnMagMat10=float,
    rlnMagMat11=float,
    # Micrograph
    rlnCoordinateX=float,
    rlnCoordinateY=float,
    rlnCoordinateZ=float,
    rlnMovieFrameNumber=int,
    rlnNormCorrection=float,
    rlnMagnificationCorrection=float,
    rlnSamplingRate=float,
    rlnSamplingRateX=float,
    rlnSamplingRateY=float,
    rlnSamplingRateZ=float,
    rlnImageSize=int,
    rlnImageSizeX=int,
    rlnImageSizeY=int,
    rlnImageSizeZ=int,
    rlnMinimumValue=float,
    rlnMaximumValue=float,
    rlnAverageValue=float,
    rlnStandardDeviationValue=float,
    rlnSkewnessValue=float,
    rlnKurtosisExcessValue=float,
    rlnImageWeight=float,
    # Mask
    rlnMaskName=object,
    # Job
    rlnJobIsContinue=bool,
    rlnJobType=int,
    rlnJobTypeName=object,
    # Job option
    rlnJoboptionType=int,
    rlnJobOptionVariable=object,
    rlnJobOptionValue=object,
    rlnJobOptionGUILabel=object,
    rlnJobOptionDefaultValue=object,
    rlnJobOptionSliderMin=float,
    rlnJobOptionSliderMax=float,
    rlnJobOptionSliderStep=float,
    rlnJobOptionHelpText=object,
    rlnJobOptionFilePattern=object,
    rlnJobOptionDirectoryDefault=object,
    rlnJobOptionMenuOptions=object,
    # Matrix
    rlnMatrix_1_1=float,
    rlnMatrix_1_2=float,
    rlnMatrix_1_3=float,
    rlnMatrix_2_1=float,
    rlnMatrix_2_2=float,
    rlnMatrix_2_3=float,
    rlnMatrix_3_1=float,
    rlnMatrix_3_2=float,
    rlnMatrix_3_3=float,
    # Motion
    rlnAccumMotionTotal=float,
    rlnAccumMotionEarly=float,
    rlnAccumMotionLate=float,
    rlnMicrographId=int,
    rlnMicrographName=object,
    rlnMicrographGainName=object,
    rlnMicrographDefectFile=object,
    rlnMicrographNameNoDW=object,
    rlnMicrographMovieName=object,
    rlnMicrographMetadata=object,
    rlnMicrographTiltAngle=float,
    rlnMicrographTiltAxisDirection=float,
    rlnMicrographTiltAxisOutOfPlane=float,
    rlnMicrographOriginalPixelSize=float,
    rlnMicrographPixelSize=float,
    rlnMicrographPreExposure=float,
    rlnMicrographDoseRate=float,
    rlnMicrographBinning=float,
    rlnMicrographFrameNumber=int,
    rlnMotionModelVersion=int,
    rlnMicrographStartFrame=int,
    rlnMicrographEndFrame=int,
    rlnMicrographShiftX=float,
    rlnMicrographShiftY=float,
    rlnMotionModelCoeffsIdx=int,
    rlnMotionModelCoeff=float,
    rlnEERUpsampling=int,
    rlnEERGrouping=int,
    # Helical
    rlnAccuracyRotations=float,
    rlnAccuracyTranslations=float,
    rlnAccuracyTranslationsAngst=float,
    rlnAveragePmax=float,
    rlnCurrentResolution=float,
    rlnCurrentImageSize=int,
    rlnSsnrMap=float,
    rlnReferenceDimensionality=int,
    rlnDataDimensionality=int,
    rlnDiff2RandomHalves=float,
    rlnEstimatedResolution=float,
    rlnFourierCompleteness=float,
    rlnOverallFourierCompleteness=float,
    rlnGoldStandardFsc=float,
    rlnGroupName=object,
    rlnGroupNumber=int,
    rlnGroupNrParticles=int,
    rlnGroupScaleCorrection=float,
    rlnNrHelicalAsymUnits=int,
    rlnHelicalTwist=float,
    rlnHelicalTwistMin=float,
    rlnHelicalTwistMax=float,
    rlnHelicalTwistInitialStep=float,
    rlnHelicalRise=float,
    rlnHelicalRiseMin=float,
    rlnHelicalRiseMax=float,
    rlnHelicalRiseInitialStep=float,
    rlnIsHelix=bool,
    rlnFourierSpaceInterpolator=int,
    rlnLogLikelihood=float,
    rlnMinRadiusNnInterpolation=int,
    rlnNormCorrectionAverage=float,
    rlnNrClasses=int,
    rlnNrBodies=int,
    rlnNrGroups=int,
    rlnSpectralOrientabilityContribution=float,
    rlnOriginalImageSize=int,
    rlnPaddingFactor=float,
    rlnClassDistribution=float,
    rlnClassPriorOffsetX=float,
    rlnClassPriorOffsetY=float,
    rlnOrientationDistribution=float,
    rlnPixelSize=float,
    rlnReferenceSpectralPower=float,
    rlnOrientationalPriorMode=int,
    rlnReferenceImage=object,
    rlnSGDGradientImage=object,
    rlnSigmaOffsets=float,
    rlnSigmaOffsetsAngst=float,
    rlnSigma2Noise=float,
    rlnReferenceSigma2=float,
    rlnSigmaPriorRotAngle=float,
    rlnSigmaPriorTiltAngle=float,
    rlnSigmaPriorPsiAngle=float,
    rlnSignalToNoiseRatio=float,
    rlnTau2FudgeFactor=float,
    rlnReferenceTau2=float,
    # Classification
    rlnOverallAccuracyRotations=float,
    rlnOverallAccuracyTranslations=float,
    rlnOverallAccuracyTranslationsAngst=float,
    rlnAdaptiveOversampleFraction=float,
    rlnAdaptiveOversampleOrder=int,
    rlnAutoLocalSearchesHealpixOrder=int,
    rlnAvailableMemory=float,
    rlnBestResolutionThusFar=float,
    rlnCoarseImageSize=int,
    rlnChangesOptimalOffsets=float,
    rlnChangesOptimalOrientations=float,
    rlnChangesOptimalClasses=float,
    rlnCtfDataArePhaseFlipped=bool,
    rlnCtfDataAreCtfPremultiplied=bool,
    rlnExperimentalDataStarFile=object,
    rlnDoCorrectCtf=bool,
    rlnDoCorrectMagnification=bool,
    rlnDoCorrectNorm=bool,
    rlnDoCorrectScale=bool,
    rlnDoExternalReconstruct=bool,
    rlnDoRealignMovies=bool,
    rlnDoMapEstimation=bool,
    rlnDoStochasticGradientDescent=bool,
    rlnDoStochasticEM=bool,
    rlnExtReconsDataReal=object,
    rlnExtReconsDataImag=object,
    rlnExtReconsWeight=object,
    rlnExtReconsResult=object,
    rlnExtReconsResultStarfile=object,
    rlnDoFastSubsetOptimisation=bool,
    rlnSgdInitialIterations=int,
    rlnSgdFinalIterations=int,
    rlnSgdInBetweenIterations=int,
    rlnSgdInitialResolution=float,
    rlnSgdFinalResolution=float,
    rlnSgdInitialSubsetSize=int,
    rlnSgdFinalSubsetSize=int,
    rlnSgdMuFactor=float,
    rlnSgdSigma2FudgeInitial=float,
    rlnSgdSigma2FudgeHalflife=int,
    rlnSgdSkipAnneal=bool,
    rlnSgdSubsetSize=int,
    rlnSgdWriteEverySubset=int,
    rlnSgdMaxSubsets=int,
    rlnSgdStepsize=float,
    rlnDoAutoRefine=bool,
    rlnDoOnlyFlipCtfPhases=bool,
    rlnDoSolventFlattening=bool,
    rlnDoSolventFscCorrection=bool,
    rlnDoSkipAlign=bool,
    rlnDoSkipRotate=bool,
    rlnDoSplitRandomHalves=bool,
    rlnDoZeroMask=bool,
    rlnFixSigmaNoiseEstimates=bool,
    rlnFixSigmaOffsetEstimates=bool,
    rlnFixTauEstimates=bool,
    rlnHasConverged=bool,
    rlnHasHighFscAtResolLimit=bool,
    rlnHasLargeSizeIncreaseIterationsAgo=int,
    rlnDoHelicalRefine=bool,
    rlnIgnoreHelicalSymmetry=bool,
    rlnFourierMask=object,
    rlnHelicalTwistInitial=float,
    rlnHelicalRiseInitial=float,
    rlnHelicalCentralProportion=float,
    rlnNrHelicalNStart=int,
    rlnHelicalMaskTubeInnerDiameter=float,
    rlnHelicalMaskTubeOuterDiameter=float,
    rlnHelicalSymmetryLocalRefinement=bool,
    rlnHelicalSigmaDistance=float,
    rlnHelicalKeepTiltPriorFixed=bool,
    rlnLowresLimitExpectation=float,
    rlnHighresLimitExpectation=float,
    rlnHighresLimitSGD=float,
    rlnDoIgnoreCtfUntilFirstPeak=bool,
    rlnIncrementImageSize=int,
    rlnCurrentIteration=int,
    rlnLocalSymmetryFile=object,
    rlnJoinHalvesUntilThisResolution=float,
    rlnMagnificationSearchRange=float,
    rlnMagnificationSearchStep=float,
    rlnMaximumCoarseImageSize=int,
    rlnMaxNumberOfPooledParticles=int,
    rlnModelStarFile=object,
    rlnModelStarFile2=object,
    rlnNumberOfIterations=int,
    rlnNumberOfIterWithoutResolutionGain=int,
    rlnNumberOfIterWithoutChangingAssignments=int,
    rlnOpticsStarFile=object,
    rlnOutputRootName=object,
    rlnParticleDiameter=float,
    rlnRadiusMaskMap=int,
    rlnRadiusMaskExpImages=int,
    rlnRandomSeed=int,
    rlnRefsAreCtfCorrected=bool,
    rlnSmallestChangesClasses=int,
    rlnSmallestChangesOffsets=float,
    rlnSmallestChangesOrientations=float,
    rlnOrientSamplingStarFile=object,
    rlnSolventMaskName=object,
    rlnSolventMask2Name=object,
    rlnTauSpectrumName=object,
    rlnUseTooCoarseSampling=bool,
    rlnWidthMaskEdge=int,
    # Orientation
    rlnIsFlip=bool,
    rlnOrientationsID=int,
    rlnOriginX=float,
    rlnOriginY=float,
    rlnOriginZ=float,
    rlnOriginXPrior=float,
    rlnOriginYPrior=float,
    rlnOriginZPrior=float,
    # Origin
    rlnOriginXAngst=float,
    rlnOriginYAngst=float,
    rlnOriginZAngst=float,
    rlnOriginXPriorAngst=float,
    rlnOriginYPriorAngst=float,
    rlnOriginZPriorAngst=float,
    # Angle
    rlnAngleRot=float,
    rlnAngleRotPrior=float,
    rlnAngleRotFlipRatio=float,
    rlnAngleTilt=float,
    rlnAngleTiltPrior=float,
    rlnAnglePsi=float,
    rlnAnglePsiPrior=float,
    rlnAnglePsiFlipRatio=float,
    rlnAnglePsiFlip=bool,
    # Picking
    rlnAutopickFigureOfMerit=float,
    rlnHelicalTubeID=int,
    rlnHelicalTubePitch=float,
    rlnHelicalTrackLength=float,
    rlnHelicalTrackLengthAngst=float,
    rlnClassNumber=int,
    rlnLogLikeliContribution=float,
    rlnParticleId=int,
    rlnParticleFigureOfMerit=float,
    rlnKullbackLeiblerDivergence=float,
    rlnKullbackLeibnerDivergence=float,  # wrong spelling for backwards compatibility
    rlnRandomSubset=int,
    rlnBeamTiltClass=int,
    rlnParticleName=object,
    rlnOriginalParticleName=object,
    rlnNrOfSignificantSamples=int,
    rlnNrOfFrames=int,
    rlnAverageNrOfFrames=int,
    rlnMovieFramesRunningAverage=int,
    rlnMaxValueProbDistribution=float,
    rlnParticleNumber=int,
    # Pipeline
    rlnPipeLineJobCounter=int,
    rlnPipeLineNodeName=object,
    rlnPipeLineNodeType=int,
    rlnPipeLineProcessAlias=object,
    rlnPipeLineProcessName=object,
    rlnPipeLineProcessType=int,
    rlnPipeLineProcessStatus=int,
    rlnPipeLineEdgeFromNode=object,
    nPipeLineEdgeToNode=object,
    nPipeLineEdgeProcess=object,
    # FSC
    rlnFinalResolution=float,
    rlnBfactorUsedForSharpening=float,
    rlnParticleBoxFractionMolecularWeight=float,
    rlnParticleBoxFractionSolventMask=float,
    rlnFourierShellCorrelation=float,
    rlnFourierShellCorrelationCorrected=float,
    rlnFourierShellCorrelationParticleMolWeight=float,
    rlnFourierShellCorrelationParticleMaskFraction=float,
    rlnFourierShellCorrelationMaskedMaps=float,
    rlnFourierShellCorrelationUnmaskedMaps=float,
    rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps=float,
    rlnAmplitudeCorrelationMaskedMaps=float,
    rlnAmplitudeCorrelationUnmaskedMaps=float,
    rlnDifferentialPhaseResidualMaskedMaps=float,
    rlnDifferentialPhaseResidualUnmaskedMaps=float,
    rlnFittedInterceptGuinierPlot=float,
    rlnFittedSlopeGuinierPlot=float,
    rlnCorrelationFitGuinierPlot=float,
    rlnLogAmplitudesOriginal=float,
    rlnLogAmplitudesMTFCorrected=float,
    rlnLogAmplitudesWeighted=float,
    rlnLogAmplitudesSharpened=float,
    rlnLogAmplitudesIntercept=float,
    rlnResolutionSquared=float,
    rlnMolecularWeight=float,
    rlnMtfValue=float,
    rlnRandomiseFrom=float,
    rlnUnfilteredMapHalf1=object,
    rlnUnfilteredMapHalf2=object,
    # Sampling
    rlnIs3DSampling=bool,
    rlnIs3DTranslationalSampling=bool,
    rlnHealpixOrder=int,
    rlnHealpixOrderOriginal=int,
    rlnTiltAngleLimit=float,
    rlnOffsetRange=float,
    rlnOffsetStep=float,
    rlnOffsetRangeOriginal=float,
    rlnOffsetStepOriginal=float,
    rlnHelicalOffsetStep=float,
    rlnSamplingPerturbInstance=float,
    rlnSamplingPerturbFactor=float,
    rlnPsiStep=float,
    rlnPsiStepOriginal=float,
    rlnSymmetryGroup=object,
    # Schedule
    rlnScheduleEdgeNumber=int,
    rlnScheduleEdgeInputNodeName=object,
    rlnScheduleEdgeOutputNodeName=object,
    rlnScheduleEdgeIsFork=bool,
    rlnScheduleEdgeOutputNodeNameIfTrue=object,
    rlnScheduleEdgeBooleanVariable=object,
    rlnScheduleCurrentNodeName=object,
    rlnScheduleOriginalStartNodeName=object,
    rlnScheduleEmailAddress=object,
    rlnScheduleName=object,
    rlnScheduleJobName=object,
    rlnScheduleJobNameOriginal=object,
    rlnScheduleJobMode=object,
    rlnScheduleJobHasStarted=bool,
    rlnScheduleOperatorName=object,
    rlnScheduleOperatorType=object,
    rlnScheduleOperatorInput1=object,
    rlnScheduleOperatorInput2=object,
    rlnScheduleOperatorOutput=object,
    rlnScheduleBooleanVariableName=object,
    rlnScheduleBooleanVariableValue=bool,
    rlnScheduleBooleanVariableResetValue=bool,
    rlnScheduleFloatVariableName=object,
    rlnScheduleFloatVariableValue=float,
    rlnScheduleFloatVariableResetValue=float,
    rlnScheduleStringVariableName=object,
    rlnScheduleStringVariableValue=object,
    rlnScheduleStringVariableResetValue=object,
    # More particles
    rlnSelected=int,
    rlnParticleSelectZScore=float,
    rlnSortedIndex=int,
    rlnStarFileMovieParticles=object,
    rlnPerFrameCumulativeWeight=float,
    rlnPerFrameRelativeWeight=float,
    # Resolution
    rlnResolution=float,
    rlnAngstromResolution=float,
    rlnResolutionInversePixel=float,
    rlnSpectralIndex=int,
    rlnUnknownLabel=object,
)


def read(file: Union[str, PurePath, IO[str]]) -> Dict[str, "NDArray"]:
    """
    Read the given STAR file into memory.

    Args:
        file (str | Path | IO): Path or file handle to ``.star`` file.

    Returns:
        dict[str, NDArray]: a dictionary of numpy record arrays

        Each key is a block name found in the star file (e.g., ``"particles"``
        for block ``data_particles`` and ``""`` for block ``data_``)

    Examples:

        Read a star file with a sole ``data_`` block

        >>> from cryosparc import star
        >>> data = star.read('particles.star')['']
        >>> data
        array([...])

        Read a star file with multiple blocks

        >>> blocks = star.read('particles_with_optics.star')
        >>> blocks['particles']
        array([...])
        >>> blocks['optics']
        array([...])
    """

    # If numpy.loadtxt has a max_rows argument, can read more efficiently.
    from inspect import signature
    import tempfile

    use_max_rows = "max_rows" in signature(n.loadtxt).parameters

    with topen(file, "r") as f:
        data_blocks: List[tuple] = []
        line = 0
        skiprows = 0

        # Stage 1: Determine available fields the STAR file and what line the
        # data starts and ends at
        while True:
            skiprows = 0
            num, val = _read_until(f, lambda x: x.strip().startswith("data_"))
            if num is None:
                assert data_blocks, f"Cannot find any 'data_' blocks in the STAR file {file}."
                break

            line += num
            skiprows += num
            dblk_name = val.strip().replace("data_", "", 1)
            dblk_start = line - 1  # which line is 'data_'

            num, val = _read_until(f, lambda x: x.strip().startswith("loop_"))
            assert (
                num is not None
            ), f"Cannot find any 'loop_' in the data block starting at line {dblk_start} in the STAR file."
            line += num
            skiprows += num

            num, val = _read_until(f, lambda x: x.strip().startswith("_"))
            assert (
                num is not None
            ), f"Cannot find start of label names in the data block starting at line {dblk_start} in the STAR file."
            line += num
            skiprows += num
            dtype: List[Tuple[str, Type[object]]] = []
            while val:
                val = val.strip()
                if val.startswith("_"):
                    label = val.lstrip("_").split(" ")[0]
                    dtype.append((label, RLN_DTYPES.get(label, object)))
                elif val.startswith("#") or val.startswith(";"):
                    pass  # comment
                else:
                    break  # done reading fields
                line += 1
                skiprows += 1
                val = f.readline()

            skiprows -= 1
            num, val = _read_until(f, lambda x: x.strip() == "", allow_eof=True)  # look for empty line or EOF
            line += num
            maxrows = num

            data_blocks.append((dblk_name, dtype, skiprows, maxrows))

        # STAGE 2: Seek back to the beginning of the file and read in rows
        f.seek(0)
        data = {}
        for dblk_name, dtype, skiprows, maxrows in data_blocks:
            if use_max_rows:
                # Can read directly
                data[dblk_name] = n.loadtxt(f, dtype=dtype, skiprows=skiprows, max_rows=maxrows, ndmin=1)
                f.readline()
                continue

            # Cannot read directly because loadtxt attempts to read every row.
            # Save required rows to tempfile first and read from there
            with tempfile.TemporaryFile("w+") as temp:
                for _ in range(skiprows):
                    f.readline()
                for _ in range(maxrows):
                    temp.write(f.readline())
                temp.seek(0)
                data[dblk_name] = n.loadtxt(temp, dtype=dtype, ndmin=1)

            f.readline()  # skip one more line

        return data


def write(
    file: Union[str, PurePath, IO[str]],
    data: Any,
    name: str = "",
    labels: Optional[List[str]] = None,
):
    """
    Write a star file with a single "data\\_" block. Data may be provided as
    either a numpy record array or a collection of tuples with a specified
    labels argument.

    Args:
        file (str | Path | IO): File path or handle to write.
        data (any): Numpy record array or Python list of tuples.
        name (str): Name of data block, to be prepended with "data\\_" when
            written to the star file. Defaults to "".
        labels (list[str], optional): Names of each column in the data. Not
            required if given a numpy record array that includes the names.
            Defaults to None.

    Examples:

        With array of tuples

        >>> from cryosparc import star
        >>> star.write('one.star', [
        ...     (123., 456.),
        ...     (789., 987.)
        ... ], labels=['rlnCoordinateX', 'rlnCoordinateY'])

        With numpy record array

        >>> arr  = np.core.records.fromrecords([
        ...     (123., 456.),
        ...     (789., 987.)
        ... ], names=[('rlnCoordinateX', 'f8') , ('rlnCoordinateY', 'f8')])
        >>> star.write('two.star', arr)
    """
    if not isinstance(data, n.ndarray):
        assert labels, f"Cannot write STAR file data with missing labels: {data}"
        names = ",".join(labels)
        data = fromrecords(data, names=names)  # type: ignore
    return write_blocks(file, {name: data})


def write_blocks(file: Union[str, PurePath, IO[str]], blocks: Mapping[str, "NDArray"]):
    """
    Write a single star file composed of multiple data blocks:

    Args:
        file (str | Path | IO): File path or handle to write.
        blocks (Mapping[str, NDArray]): Dictionary of record arrays to write.

    Examples:

        With optics group and particles

        >>> from cryosparc import star
        >>> import numpy as np
        >>> optics = np.core.records.fromrecords([
        ...     ('mydata', ..., 0.1, 0.1)
        ... ], names='rlnOpticsGroupName,...,rlnBeamTiltX,rlnBeamTiltY'])
        >>> particles = np.core.records.fromrecords([
        ...     (123., 456.), ... (789., 987.),
        ... ], names='rlnCoordinateX,rlnCoordinateY')
        >>> star.write('particles.star', {
        ...     'optics': optics,
        ...     'particles': particles
        ... })
    """
    # Check that each value has labels to write
    entries = []
    for k, d in blocks.items():
        # Infer labels from numpy record array
        labels = [f[0] for f in d.dtype.descr]
        assert all(labels), f"Cannot write STAR file data with missing labels: {d}"
        entries.append((k, d, labels))

    with topen(file, "w") as f:
        for name, block, labels in entries:
            f.writelines(["\n", f"data_{name}\n", "\n", "loop_\n"])
            f.writelines(f"_{field} #{i + 1}\n" for i, field in enumerate(labels))
            for row in block:
                f.writelines([" ".join(map(str, row)), "\n"])
            f.write("\n")


@overload
def _read_until(f: IO[str], line_test: Callable[[str], bool]) -> Tuple[Optional[int], str]:
    ...


@overload
def _read_until(f: IO[str], line_test: Callable[[str], bool], allow_eof: Literal[True]) -> Tuple[int, str]:
    ...


def _read_until(f: IO[str], line_test: Callable[[str], bool], allow_eof=False) -> Tuple[Optional[int], str]:
    # Read from the given file handle line-by-line until the line m
    num_lines = 0
    inp = ""
    while True:
        num_lines += 1
        inp = f.readline()
        if inp == "":  # end of file
            return (num_lines if allow_eof else None), inp
        elif line_test(inp):  # found the test condition
            return num_lines, inp
