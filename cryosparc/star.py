from pathlib import PurePath
from typing import IO, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, overload
from typing_extensions import Literal
import numpy as n
import numpy.typing as nt
from numpy.core.records import fromrecords

from .util import topen

Fmt = Union[str, Sequence[str]]

# Type definitions:
# O = string
# i4 = integer
# f8 = double
# u1 = bool
RLN_DTYPES = dict(
    rlnComment="O",
    # Area
    rlnAreaId="i4",
    rlnAreaName="O",
    # Body
    rlnBodyMaskName="O",
    rlnBodyKeepFixed="i4",
    rlnBodyReferenceName="O",
    rlnBodyRotateDirectionX="f8",
    rlnBodyRotateDirectionY="f8",
    rlnBodyRotateDirectionZ="f8",
    rlnBodyRotateRelativeTo="i4",
    rlnBodySigmaAngles="f8",
    rlnBodySigmaOffset="f8",
    rlnBodySigmaOffsetAngst="f8",
    rlnBodySigmaRot="f8",
    rlnBodySigmaTilt="f8",
    rlnBodySigmaPsi="f8",
    rlnBodyStarFile="O",
    # CTF
    rlnCtfAstigmatism="f8",
    rlnCtfBfactor="f8",
    rlnCtfMaxResolution="f8",
    rlnCtfValidationScore="f8",
    rlnCtfScalefactor="f8",
    rlnVoltage="f8",
    rlnDefocusU="f8",
    rlnDefocusV="f8",
    rlnDefocusAngle="f8",
    rlnSphericalAberration="f8",
    rlnChromaticAberration="f8",
    rlnDetectorPixelSize="f8",
    rlnCtfPowerSpectrum="O",
    rlnEnergyLoss="f8",
    rlnCtfFigureOfMerit="f8",
    rlnCtfImage="O",
    rlnLensStability="f8",
    rlnMagnification="f8",
    rlnPhaseShift="f8",
    rlnConvergenceCone="f8",
    rlnLongitudinalDisplacement="f8",
    rlnTransversalDisplacement="f8",
    rlnAmplitudeContrast="f8",
    rlnCtfValue="f8",
    # Image
    rlnImageName="O",
    rlnImageOriginalName="O",
    rlnReconstructImageName="O",
    rlnImageId="i4",
    rlnEnabled="u1",
    rlnDataType="i4",
    rlnImageDimensionality="i4",
    rlnBeamTiltX="f8",
    rlnBeamTiltY="f8",
    rlnMtfFileName="O",
    rlnOpticsGroup="i4",
    rlnOpticsGroupName="O",
    # rlnOddZernike='f8 VECTOR',  # TODO: Determine correct interpretation
    # rlnEvenZernike='f8 VECTOR',  # TODO: Determine correct interpretation
    rlnImagePixelSize="f8",
    rlnMagMat00="f8",
    rlnMagMat01="f8",
    rlnMagMat10="f8",
    rlnMagMat11="f8",
    # Micrograph
    rlnCoordinateX="f8",
    rlnCoordinateY="f8",
    rlnCoordinateZ="f8",
    rlnMovieFrameNumber="i4",
    rlnNormCorrection="f8",
    rlnMagnificationCorrection="f8",
    rlnSamplingRate="f8",
    rlnSamplingRateX="f8",
    rlnSamplingRateY="f8",
    rlnSamplingRateZ="f8",
    rlnImageSize="i4",
    rlnImageSizeX="i4",
    rlnImageSizeY="i4",
    rlnImageSizeZ="i4",
    rlnMinimumValue="f8",
    rlnMaximumValue="f8",
    rlnAverageValue="f8",
    rlnStandardDeviationValue="f8",
    rlnSkewnessValue="f8",
    rlnKurtosisExcessValue="f8",
    rlnImageWeight="f8",
    # Mask
    rlnMaskName="O",
    # Job
    rlnJobIsContinue="u1",
    rlnJobType="i4",
    rlnJobTypeName="O",
    # Job option
    rlnJoboptionType="i4",
    rlnJobOptionVariable="O",
    rlnJobOptionValue="O",
    rlnJobOptionGUILabel="O",
    rlnJobOptionDefaultValue="O",
    rlnJobOptionSliderMin="f8",
    rlnJobOptionSliderMax="f8",
    rlnJobOptionSliderStep="f8",
    rlnJobOptionHelpText="O",
    rlnJobOptionFilePattern="O",
    rlnJobOptionDirectoryDefault="O",
    rlnJobOptionMenuOptions="O",
    # Matrix
    rlnMatrix_1_1="f8",
    rlnMatrix_1_2="f8",
    rlnMatrix_1_3="f8",
    rlnMatrix_2_1="f8",
    rlnMatrix_2_2="f8",
    rlnMatrix_2_3="f8",
    rlnMatrix_3_1="f8",
    rlnMatrix_3_2="f8",
    rlnMatrix_3_3="f8",
    # Motion
    rlnAccumMotionTotal="f8",
    rlnAccumMotionEarly="f8",
    rlnAccumMotionLate="f8",
    rlnMicrographId="i4",
    rlnMicrographName="O",
    rlnMicrographGainName="O",
    rlnMicrographDefectFile="O",
    rlnMicrographNameNoDW="O",
    rlnMicrographMovieName="O",
    rlnMicrographMetadata="O",
    rlnMicrographTiltAngle="f8",
    rlnMicrographTiltAxisDirection="f8",
    rlnMicrographTiltAxisOutOfPlane="f8",
    rlnMicrographOriginalPixelSize="f8",
    rlnMicrographPixelSize="f8",
    rlnMicrographPreExposure="f8",
    rlnMicrographDoseRate="f8",
    rlnMicrographBinning="f8",
    rlnMicrographFrameNumber="i4",
    rlnMotionModelVersion="i4",
    rlnMicrographStartFrame="i4",
    rlnMicrographEndFrame="i4",
    rlnMicrographShiftX="f8",
    rlnMicrographShiftY="f8",
    rlnMotionModelCoeffsIdx="i4",
    rlnMotionModelCoeff="f8",
    rlnEERUpsampling="i4",
    rlnEERGrouping="i4",
    # Helical
    rlnAccuracyRotations="f8",
    rlnAccuracyTranslations="f8",
    rlnAccuracyTranslationsAngst="f8",
    rlnAveragePmax="f8",
    rlnCurrentResolution="f8",
    rlnCurrentImageSize="i4",
    rlnSsnrMap="f8",
    rlnReferenceDimensionality="i4",
    rlnDataDimensionality="i4",
    rlnDiff2RandomHalves="f8",
    rlnEstimatedResolution="f8",
    rlnFourierCompleteness="f8",
    rlnOverallFourierCompleteness="f8",
    rlnGoldStandardFsc="f8",
    rlnGroupName="O",
    rlnGroupNumber="i4",
    rlnGroupNrParticles="i4",
    rlnGroupScaleCorrection="f8",
    rlnNrHelicalAsymUnits="i4",
    rlnHelicalTwist="f8",
    rlnHelicalTwistMin="f8",
    rlnHelicalTwistMax="f8",
    rlnHelicalTwistInitialStep="f8",
    rlnHelicalRise="f8",
    rlnHelicalRiseMin="f8",
    rlnHelicalRiseMax="f8",
    rlnHelicalRiseInitialStep="f8",
    rlnIsHelix="u1",
    rlnFourierSpaceInterpolator="i4",
    rlnLogLikelihood="f8",
    rlnMinRadiusNnInterpolation="i4",
    rlnNormCorrectionAverage="f8",
    rlnNrClasses="i4",
    rlnNrBodies="i4",
    rlnNrGroups="i4",
    rlnSpectralOrientabilityContribution="f8",
    rlnOriginalImageSize="i4",
    rlnPaddingFactor="f8",
    rlnClassDistribution="f8",
    rlnClassPriorOffsetX="f8",
    rlnClassPriorOffsetY="f8",
    rlnOrientationDistribution="f8",
    rlnPixelSize="f8",
    rlnReferenceSpectralPower="f8",
    rlnOrientationalPriorMode="i4",
    rlnReferenceImage="O",
    rlnSGDGradientImage="O",
    rlnSigmaOffsets="f8",
    rlnSigmaOffsetsAngst="f8",
    rlnSigma2Noise="f8",
    rlnReferenceSigma2="f8",
    rlnSigmaPriorRotAngle="f8",
    rlnSigmaPriorTiltAngle="f8",
    rlnSigmaPriorPsiAngle="f8",
    rlnSignalToNoiseRatio="f8",
    rlnTau2FudgeFactor="f8",
    rlnReferenceTau2="f8",
    # Classification
    rlnOverallAccuracyRotations="f8",
    rlnOverallAccuracyTranslations="f8",
    rlnOverallAccuracyTranslationsAngst="f8",
    rlnAdaptiveOversampleFraction="f8",
    rlnAdaptiveOversampleOrder="i4",
    rlnAutoLocalSearchesHealpixOrder="i4",
    rlnAvailableMemory="f8",
    rlnBestResolutionThusFar="f8",
    rlnCoarseImageSize="i4",
    rlnChangesOptimalOffsets="f8",
    rlnChangesOptimalOrientations="f8",
    rlnChangesOptimalClasses="f8",
    rlnCtfDataArePhaseFlipped="u1",
    rlnCtfDataAreCtfPremultiplied="u1",
    rlnExperimentalDataStarFile="O",
    rlnDoCorrectCtf="u1",
    rlnDoCorrectMagnification="u1",
    rlnDoCorrectNorm="u1",
    rlnDoCorrectScale="u1",
    rlnDoExternalReconstruct="u1",
    rlnDoRealignMovies="u1",
    rlnDoMapEstimation="u1",
    rlnDoStochasticGradientDescent="u1",
    rlnDoStochasticEM="u1",
    rlnExtReconsDataReal="O",
    rlnExtReconsDataImag="O",
    rlnExtReconsWeight="O",
    rlnExtReconsResult="O",
    rlnExtReconsResultStarfile="O",
    rlnDoFastSubsetOptimisation="u1",
    rlnSgdInitialIterations="i4",
    rlnSgdFinalIterations="i4",
    rlnSgdInBetweenIterations="i4",
    rlnSgdInitialResolution="f8",
    rlnSgdFinalResolution="f8",
    rlnSgdInitialSubsetSize="i4",
    rlnSgdFinalSubsetSize="i4",
    rlnSgdMuFactor="f8",
    rlnSgdSigma2FudgeInitial="f8",
    rlnSgdSigma2FudgeHalflife="i4",
    rlnSgdSkipAnneal="u1",
    rlnSgdSubsetSize="i4",
    rlnSgdWriteEverySubset="i4",
    rlnSgdMaxSubsets="i4",
    rlnSgdStepsize="f8",
    rlnDoAutoRefine="u1",
    rlnDoOnlyFlipCtfPhases="u1",
    rlnDoSolventFlattening="u1",
    rlnDoSolventFscCorrection="u1",
    rlnDoSkipAlign="u1",
    rlnDoSkipRotate="u1",
    rlnDoSplitRandomHalves="u1",
    rlnDoZeroMask="u1",
    rlnFixSigmaNoiseEstimates="u1",
    rlnFixSigmaOffsetEstimates="u1",
    rlnFixTauEstimates="u1",
    rlnHasConverged="u1",
    rlnHasHighFscAtResolLimit="u1",
    rlnHasLargeSizeIncreaseIterationsAgo="i4",
    rlnDoHelicalRefine="u1",
    rlnIgnoreHelicalSymmetry="u1",
    rlnFourierMask="O",
    rlnHelicalTwistInitial="f8",
    rlnHelicalRiseInitial="f8",
    rlnHelicalCentralProportion="f8",
    rlnNrHelicalNStart="i4",
    rlnHelicalMaskTubeInnerDiameter="f8",
    rlnHelicalMaskTubeOuterDiameter="f8",
    rlnHelicalSymmetryLocalRefinement="u1",
    rlnHelicalSigmaDistance="f8",
    rlnHelicalKeepTiltPriorFixed="u1",
    rlnLowresLimitExpectation="f8",
    rlnHighresLimitExpectation="f8",
    rlnHighresLimitSGD="f8",
    rlnDoIgnoreCtfUntilFirstPeak="u1",
    rlnIncrementImageSize="i4",
    rlnCurrentIteration="i4",
    rlnLocalSymmetryFile="O",
    rlnJoinHalvesUntilThisResolution="f8",
    rlnMagnificationSearchRange="f8",
    rlnMagnificationSearchStep="f8",
    rlnMaximumCoarseImageSize="i4",
    rlnMaxNumberOfPooledParticles="i4",
    rlnModelStarFile="O",
    rlnModelStarFile2="O",
    rlnNumberOfIterations="i4",
    rlnNumberOfIterWithoutResolutionGain="i4",
    rlnNumberOfIterWithoutChangingAssignments="i4",
    rlnOpticsStarFile="O",
    rlnOutputRootName="O",
    rlnParticleDiameter="f8",
    rlnRadiusMaskMap="i4",
    rlnRadiusMaskExpImages="i4",
    rlnRandomSeed="i4",
    rlnRefsAreCtfCorrected="u1",
    rlnSmallestChangesClasses="i4",
    rlnSmallestChangesOffsets="f8",
    rlnSmallestChangesOrientations="f8",
    rlnOrientSamplingStarFile="O",
    rlnSolventMaskName="O",
    rlnSolventMask2Name="O",
    rlnTauSpectrumName="O",
    rlnUseTooCoarseSampling="u1",
    rlnWidthMaskEdge="i4",
    # Orieantation
    rlnIsFlip="u1",
    rlnOrientationsID="i4",
    rlnOriginX="f8",
    rlnOriginY="f8",
    rlnOriginZ="f8",
    rlnOriginXPrior="f8",
    rlnOriginYPrior="f8",
    rlnOriginZPrior="f8",
    # Origin
    rlnOriginXAngst="f8",
    rlnOriginYAngst="f8",
    rlnOriginZAngst="f8",
    rlnOriginXPriorAngst="f8",
    rlnOriginYPriorAngst="f8",
    rlnOriginZPriorAngst="f8",
    # Angle
    rlnAngleRot="f8",
    rlnAngleRotPrior="f8",
    rlnAngleRotFlipRatio="f8",
    rlnAngleTilt="f8",
    rlnAngleTiltPrior="f8",
    rlnAnglePsi="f8",
    rlnAnglePsiPrior="f8",
    rlnAnglePsiFlipRatio="f8",
    rlnAnglePsiFlip="u1",
    # Picking
    rlnAutopickFigureOfMerit="f8",
    rlnHelicalTubeID="i4",
    rlnHelicalTubePitch="f8",
    rlnHelicalTrackLength="f8",
    rlnHelicalTrackLengthAngst="f8",
    rlnClassNumber="i4",
    rlnLogLikeliContribution="f8",
    rlnParticleId="i4",
    rlnParticleFigureOfMerit="f8",
    rlnKullbackLeiblerDivergence="f8",
    rlnKullbackLeibnerDivergence="f8",  # wrong spelling for backwards compatibility
    rlnRandomSubset="i4",
    rlnBeamTiltClass="i4",
    rlnParticleName="O",
    rlnOriginalParticleName="O",
    rlnNrOfSignificantSamples="i4",
    rlnNrOfFrames="i4",
    rlnAverageNrOfFrames="i4",
    rlnMovieFramesRunningAverage="i4",
    rlnMaxValueProbDistribution="f8",
    rlnParticleNumber="i4",
    # Pipeline
    rlnPipeLineJobCounter="i4",
    rlnPipeLineNodeName="O",
    rlnPipeLineNodeType="i4",
    rlnPipeLineProcessAlias="O",
    rlnPipeLineProcessName="O",
    rlnPipeLineProcessType="i4",
    rlnPipeLineProcessStatus="i4",
    rlnPipeLineEdgeFromNode="O",
    nPipeLineEdgeToNode="O",
    nPipeLineEdgeProcess="O",
    # FSC
    rlnFinalResolution="f8",
    rlnBfactorUsedForSharpening="f8",
    rlnParticleBoxFractionMolecularWeight="f8",
    rlnParticleBoxFractionSolventMask="f8",
    rlnFourierShellCorrelation="f8",
    rlnFourierShellCorrelationCorrected="f8",
    rlnFourierShellCorrelationParticleMolWeight="f8",
    rlnFourierShellCorrelationParticleMaskFraction="f8",
    rlnFourierShellCorrelationMaskedMaps="f8",
    rlnFourierShellCorrelationUnmaskedMaps="f8",
    rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps="f8",
    rlnAmplitudeCorrelationMaskedMaps="f8",
    rlnAmplitudeCorrelationUnmaskedMaps="f8",
    rlnDifferentialPhaseResidualMaskedMaps="f8",
    rlnDifferentialPhaseResidualUnmaskedMaps="f8",
    rlnFittedInterceptGuinierPlot="f8",
    rlnFittedSlopeGuinierPlot="f8",
    rlnCorrelationFitGuinierPlot="f8",
    rlnLogAmplitudesOriginal="f8",
    rlnLogAmplitudesMTFCorrected="f8",
    rlnLogAmplitudesWeighted="f8",
    rlnLogAmplitudesSharpened="f8",
    rlnLogAmplitudesIntercept="f8",
    rlnResolutionSquared="f8",
    rlnMolecularWeight="f8",
    rlnMtfValue="f8",
    rlnRandomiseFrom="f8",
    rlnUnfilteredMapHalf1="O",
    rlnUnfilteredMapHalf2="O",
    # Sampling
    rlnIs3DSampling="u1",
    rlnIs3DTranslationalSampling="u1",
    rlnHealpixOrder="i4",
    rlnHealpixOrderOriginal="i4",
    rlnTiltAngleLimit="f8",
    rlnOffsetRange="f8",
    rlnOffsetStep="f8",
    rlnOffsetRangeOriginal="f8",
    rlnOffsetStepOriginal="f8",
    rlnHelicalOffsetStep="f8",
    rlnSamplingPerturbInstance="f8",
    rlnSamplingPerturbFactor="f8",
    rlnPsiStep="f8",
    rlnPsiStepOriginal="f8",
    rlnSymmetryGroup="O",
    # Schedule
    rlnScheduleEdgeNumber="i4",
    rlnScheduleEdgeInputNodeName="O",
    rlnScheduleEdgeOutputNodeName="O",
    rlnScheduleEdgeIsFork="u1",
    rlnScheduleEdgeOutputNodeNameIfTrue="O",
    rlnScheduleEdgeBooleanVariable="O",
    rlnScheduleCurrentNodeName="O",
    rlnScheduleOriginalStartNodeName="O",
    rlnScheduleEmailAddress="O",
    rlnScheduleName="O",
    rlnScheduleJobName="O",
    rlnScheduleJobNameOriginal="O",
    rlnScheduleJobMode="O",
    rlnScheduleJobHasStarted="u1",
    rlnScheduleOperatorName="O",
    rlnScheduleOperatorType="O",
    rlnScheduleOperatorInput1="O",
    rlnScheduleOperatorInput2="O",
    rlnScheduleOperatorOutput="O",
    rlnScheduleBooleanVariableName="O",
    rlnScheduleBooleanVariableValue="u1",
    rlnScheduleBooleanVariableResetValue="u1",
    rlnScheduleFloatVariableName="O",
    rlnScheduleFloatVariableValue="f8",
    rlnScheduleFloatVariableResetValue="f8",
    rlnScheduleStringVariableName="O",
    rlnScheduleStringVariableValue="O",
    rlnScheduleStringVariableResetValue="O",
    # More particles
    rlnSelected="i4",
    rlnParticleSelectZScore="f8",
    rlnSortedIndex="i4",
    rlnStarFileMovieParticles="O",
    rlnPerFrameCumulativeWeight="f8",
    rlnPerFrameRelativeWeight="f8",
    # Resolution
    rlnResolution="f8",
    rlnAngstromResolution="f8",
    rlnResolutionInversePixel="f8",
    rlnSpectralIndex="i4",
    rlnUnknownLabel="O",
)


def read(file: Union[str, PurePath, IO[str]]) -> Dict[str, nt.NDArray]:
    """
    Read the given STAR file into memory.

    Returns a dictionary where each key is a block name found in the star file
    (e.g., `"particles"` for block `data_particles` and `""` for block `data_`)
    and each value is a numpy record array of the contents of that block.
    """
    with topen(file, "r") as f:
        data_blocks = []
        line = 0

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
            dtype = []
            while val:
                val = val.strip()
                if val.startswith("_"):
                    label = val.lstrip("_").split(" ")[0]
                    dtype.append((label, RLN_DTYPES.get(label, "S")))
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
            data[dblk_name] = n.loadtxt(f, dtype=dtype, skiprows=skiprows, max_rows=maxrows, ndmin=1)
            f.readline()  # skip one more line

        return data


def write(
    file: Union[str, PurePath, IO[str]],
    data: Any,
    name: str = "",
    labels: Optional[List[str]] = None,
):
    """
    Write a star file with a single `data_` block. Data may be provided as
    either a numpy record array or a collection of tuples with a specified
    labels argument.

    Example:

        from cryosparc import star

        star.write('one.star', [
            (123., 456.),
            (789., 123.)
        ], labels=['rlnCoordinateX', 'rlnCoordinateY'])

        arr  = np.core.records.fromrecords([
            (123., 456.),
            (789., 123.)
        ], names='rlnCoordinateX', 'f8') , ('rlnCoordinateY', 'f8')])
        star.write('two.star', arr)
    """
    if not isinstance(data, n.ndarray):
        assert labels, f"Cannot write STAR file data with missing labels: {data}"
        names = ",".join(labels)
        data = fromrecords(data, names=names)  # type: ignore
    return write_blocks(file, {name: data})


def write_blocks(file: Union[str, PurePath, IO[str]], blocks: Mapping[str, nt.NDArray]):
    """
    Write a single star file composed of multiple data blocks:

    Example optics group + particles:

        from cryosparc import star
        import numpy as np

        optics = np.core.records.fromrecords([
            ('mydata', ..., 0.1, 0.1)
        ], names='rlnOpticsGroupName,...,rlnBeamTiltX,rlnBeamTiltY'])

        particles = np.core.records.fromrecords([
            (123., 456.), ... (789., 123.),
        ], names='rlnCoordinateX,rlnCoordinateY')

        star.write('particles.star', {
            'optics': optics,
            'particles': particles
        }
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
    """
    Read from the given file handle line-by-line until the line m
    """
    num_lines = 0
    inp = ""
    while True:
        num_lines += 1
        inp = f.readline()
        if inp == "":  # end of file
            if not allow_eof:
                num_lines = None
            break
        elif line_test(inp):
            # found the test condition
            break

    return num_lines, inp
