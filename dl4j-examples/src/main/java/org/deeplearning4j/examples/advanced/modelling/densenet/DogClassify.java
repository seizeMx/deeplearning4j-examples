/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.densenet;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.advanced.modelling.densenet.imageUtils.BlurTransform;
import org.deeplearning4j.examples.advanced.modelling.densenet.imageUtils.NoiseTransform;
import org.deeplearning4j.examples.advanced.modelling.densenet.model.DenseNetModel;
import org.deeplearning4j.examples.utils.DownloaderUtility;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.*;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class DogClassify {
    private static final Logger log = LoggerFactory.getLogger(DogClassify.class);

//    private static final String MODEL_PATH = "/home/sylar/ide/ideaIC-2020.2/dog-classifier-models";
    private static final String MODEL_PATH = "/data/home/sylar/workspace/exp/dog-classifier-models";

    private static InputSplit trainingData;
    private static InputSplit validationData;

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int batchSize = 32;
    private static final int outputNum = 5;
    private static final int numEpochs = 60;
    private static final double splitTrainTest = 0.8;

    public static String dataLocalPath;

    private static final String featureExtractionLayer = "fc2";

    protected static final int numClasses = 5;

    public static void main(String[] args) throws Exception {
//        File mainPath = new File("/home/sylar/ide/ideaIC-2020.2/dog-classifier");
        File mainPath = new File("/data/home/sylar/workspace/exp/imgs");
        Random random = new Random(1234);

        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
//        DataNormalization dataNormalization = new ImagePreProcessingScaler(0, 1);
        DataNormalization dataNormalization = new VGG16ImagePreProcessor();

        BalancedPathFilter pathFilter = new BalancedPathFilter(random, labelMaker, 0);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        trainingData = inputSplit[0];
        validationData = inputSplit[1];


        ComputationGraph computationGraph = getComputationGraphVGG();


        setListeners(computationGraph, dataNormalization, labelMaker, 1);

        log.info("TRAIN MODEL");
        trainData(dataNormalization, labelMaker, computationGraph);
    }

//https://dl4jdata.blob.core.windows.net/models/vgg16_dl4j_inference.zip
    private static ComputationGraph getComputationGraphRES() throws IOException {
        log.info("BUILD MODEL");
        ZooModel zooModel = NASNet.builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
//            .inputShape(new int[]{3, 224, 224})
            .numClasses(5)
            .build();
//        zooModel.setInputShape(new int[][]{{3, 224, 224}});
        ComputationGraph sq  = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(sq.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nadam())
            .seed(1234)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph computationGraph = new TransferLearning.GraphBuilder(sq)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor("global_average_pooling2d_3") //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1056).nOut(5)
                    .weightInit(WeightInit.XAVIER) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "global_average_pooling2d_3")
            .setInputTypes(InputType.convolutional(height, width, channels))
            .build();
        log.info(computationGraph.summary());


//        ComputationGraph computationGraph = DenseNetModel.getInstance().buildNetwork(432545609, channels, outputNum, width, height);
        return computationGraph;
    }

    private static ComputationGraph getComputationGraphSQ() throws IOException {
        log.info("BUILD MODEL");
        ZooModel zooModel = SqueezeNet.builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
//            .inputShape(new int[]{3, 224, 224})
            .numClasses(5)
            .build();
//        zooModel.setInputShape(new int[][]{{3, 224, 224}});
        ComputationGraph sq  = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(sq.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nadam())
            .seed(1234)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph computationGraph = new TransferLearning.GraphBuilder(sq)
            .fineTuneConfiguration(fineTuneConf)
//            .setFeatureExtractor("drop9") //the specified layer and below are "frozen"
//            .removeVertexKeepConnections("conv10") //replace the functionality of the final vertex
//            .addLayer("conv10",
//                new ConvolutionLayer.Builder(1,1).nOut(5)
//                    .weightInit(WeightInit.XAVIER) //This weight init dist gave better results than Xavier
//                    .build(),
//                "drop9")
            .setInputTypes(InputType.convolutional(height, width, channels))
            .setOutputs("global_average_pooling2d_5")
            .build();
        log.info(computationGraph.summary());


//        ComputationGraph computationGraph = DenseNetModel.getInstance().buildNetwork(432545609, channels, outputNum, width, height);
        return computationGraph;
    }


    private static ComputationGraph getComputationGraphSQ_CU() throws IOException {
        log.info("BUILD MODEL");

        ZooModel zooModel = SqueezeNet.builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            .updater(new Adam(new StepSchedule(ScheduleType.EPOCH, 0.001D, 0.5, 20)))
            .weightInit( WeightInit.XAVIER)
            .numClasses(5)
            .build();

        ComputationGraph sq  = zooModel.init();
        log.info(sq.summary());

        return sq;
    }


    private static ComputationGraph getComputationGraphVGG() throws IOException {
        log.info("BUILD MODEL");

        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Nesterovs(5e-5))
            .seed(1234)
            .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .fineTuneConfiguration(fineTuneConf)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(4096).nOut(numClasses)
                    .weightInit(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                    .activation(Activation.SOFTMAX).build(),
                "fc2")
            .build();
        log.info(vgg16Transfer.summary());

        return vgg16Transfer;
    }


    private static ImageTransform getImageTransform() {
        Random random = new Random(1234);
//        ImageTransform RandomCrop = new RandomCropTransform(height, width);
        ImageTransform RandomCrop = new FlipImageTransform(0);
        ImageTransform show = new ShowImageTransform("Display");

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
            new Pair<>(RandomCrop, 0.5)/*,
            new Pair<>(show, 1.0)*/
        );
        return new PipelineImageTransform(pipeline, false);
    }

    private static void trainData(DataNormalization dataNormalization, ParentPathLabelGenerator labelMaker, ComputationGraph computationGraph) {
        try {
            ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
            recordReader.initialize(trainingData, getImageTransform());
            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
            dataNormalization.fit(dataSetIterator);
            dataSetIterator.setPreProcessor(dataNormalization);
            computationGraph.fit(dataSetIterator, numEpochs);
            dataSetIterator.reset();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void setListeners(ComputationGraph computationGraph, DataNormalization dataNormalization, ParentPathLabelGenerator labelMaker, int epochs) {
        try {
            UIServer uiServer = UIServer.getInstance();

            ImageRecordReader trainingRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
            trainingRecordReader.initialize(trainingData, getImageTransform());
            DataSetIterator trainingDataSetIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, 1, outputNum);
            dataNormalization.fit(trainingDataSetIterator);
            trainingDataSetIterator.setPreProcessor(dataNormalization);
            EvaluativeListener evaluativeTrainingListener = new EvaluativeListener(trainingDataSetIterator, epochs, InvocationType.EPOCH_END, new Evaluation(outputNum));

            ImageRecordReader validationRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
            validationRecordReader.initialize(validationData, null);
            DataSetIterator validationDataSetIterator = new RecordReaderDataSetIterator(validationRecordReader, batchSize, 1, outputNum);
            dataNormalization.fit(validationDataSetIterator);
            validationDataSetIterator.setPreProcessor(dataNormalization);
            EvaluativeListener evaluativeValidationListener = new EvaluativeListener(validationDataSetIterator, epochs, InvocationType.EPOCH_END, new Evaluation(outputNum));

            StatsStorage statsStorage = new InMemoryStatsStorage();
            StatsListener statsListener = new StatsListener(statsStorage);

            ScoreIterationListener scoreIterationListener = new ScoreIterationListener(1);

            File model = new File(MODEL_PATH);
            boolean newRun = false;
            if (!model.exists()) {
                newRun = model.mkdir();
            }
            CheckpointListener checkpointListener = new CheckpointListener.Builder(model)
                .keepAll()
                .deleteExisting(!newRun)
                .saveEveryNEpochs(epochs)
                .build();

            uiServer.attach(statsStorage);
            computationGraph.setListeners(evaluativeValidationListener, statsListener, scoreIterationListener, checkpointListener);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
