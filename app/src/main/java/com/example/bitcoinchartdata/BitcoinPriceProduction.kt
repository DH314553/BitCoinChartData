package com.example.bitcoinchartdata

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.lossfunctions.LossFunctions

class BitcoinPriceProduction {
    private val V_HEIGHT = 13
    private val V_WIDTH = 4
    private val kernelSize = 2
    private val numChannels = 1
    private val numSkipLines = 1
    private val regression = true
    private val batchSize = 32
    var trainFeatures = CSVSequenceRecordReader(numSkipLines, ",")
    var trainTargets = CSVSequenceRecordReader(numSkipLines, ",")
    var train = SequenceRecordReaderDataSetIterator(
        trainFeatures, trainTargets, batchSize,
        10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH
    )
    var testFeatures = CSVSequenceRecordReader(numSkipLines, ",")
    var testTargets = CSVSequenceRecordReader(numSkipLines, ",")
    var test = SequenceRecordReaderDataSetIterator(
        testFeatures, testTargets, batchSize,
        10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH
    )
    var conf = NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .seed(12345)
        .weightInit(WeightInit.XAVIER)
        .updater(AdaGrad(0.005))
        .list()
        .layer(
            0, ConvolutionLayer.Builder(kernelSize, kernelSize)
                .nIn(1) //1 channel
                .nOut(7)
                .stride(2, 2)
                .activation(Activation.RELU)
                .build()
        )
        .layer(
            1, LSTM.Builder()
                .activation(Activation.SOFTSIGN)
                .nIn(84)
                .nOut(200)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10.0)
                .build()
        )
        .layer(
            2, RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(200)
                .nOut(52)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10.0)
                .build()
        )
        .inputPreProcessor(0, RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
        .inputPreProcessor(1, CnnToRnnPreProcessor(6, 2, 7))
        .build()
    var net = MultiLayerNetwork(conf)
}