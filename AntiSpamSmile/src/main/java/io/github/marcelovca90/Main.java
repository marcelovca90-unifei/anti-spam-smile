package io.github.marcelovca90;

import java.awt.BorderLayout;
import java.awt.Color;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.apache.commons.lang3.tuple.Pair;

import smile.classification.DecisionTree;
import smile.classification.KNN;
import smile.classification.LDA;
import smile.classification.NeuralNetwork;
import smile.classification.NeuralNetwork.ErrorFunction;
import smile.classification.QDA;
import smile.classification.RBFNetwork;
import smile.classification.RandomForest;
import smile.classification.SVM;
import smile.data.AttributeDataset;
import smile.math.Math;
import smile.math.distance.EuclideanDistance;
import smile.math.kernel.GaussianKernel;
import smile.math.rbf.GaussianRadialBasis;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

public class Main
{
    enum ClassType
    {
        HAM, SPAM
    }

    public static void main(String[] args) throws Exception
    {
        final String filenameHam = "C:\\Users\\marcelovca90\\git\\anti-spam-weka-data\\2017_BASE2\\2017_BASE2_SPAM_ASSASSIN\\MI\\8\\ham";
        final String filenameSpam = "C:\\Users\\marcelovca90\\git\\anti-spam-weka-data\\2017_BASE2\\2017_BASE2_SPAM_ASSASSIN\\MI\\8\\spam";

        // read data
        AttributeDataset ham = DataSetHelper.read(filenameHam, ClassType.HAM);
        AttributeDataset spam = DataSetHelper.read(filenameSpam, ClassType.SPAM);
        AttributeDataset dataSet = DataSetHelper.mergeDataSets(ham, spam);

        // shuffle data
        dataSet = DataSetHelper.shuffle(dataSet, 0);

        // build train/test data
        Pair<AttributeDataset, AttributeDataset> pair = DataSetHelper.split(dataSet, 0.5);
        AttributeDataset train = pair.getLeft();
        AttributeDataset test = pair.getRight();

        double[][] trainx = train.toArray(new double[train.size()][]);
        int[] trainy = train.toArray(new int[train.size()]);
        double[][] testx = test.toArray(new double[test.size()][]);
        int[] testy = test.toArray(new int[test.size()]);

        // probability
        runLda(trainx, trainy, testx, testy);
        runQda(trainx, trainy, testx, testy);

        // trees and forests
        runDt(trainx, trainy, testx, testy);
        runRf(trainx, trainy, testx, testy);

        // neural networks
        runMlp(trainx, trainy, testx, testy);
        runRbf(trainx, trainy, testx, testy);

        // others
        runSvm(trainx, trainy, testx, testy);
        runKnn(trainx, trainy, testx, testy);
    }

    private static void runLda(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        LDA lda = new LDA(trainx, trainy, 1e-9);

        int errorLda = 0;
        for (int i = 0; i < testx.length; i++)
            if (lda.predict(testx[i]) != testy[i])
                errorLda++;

        System.out.format("LDA Error rate = %.2f%%\n", 100.0 * errorLda / testx.length);
    }

    private static void runQda(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        QDA qda = new QDA(trainx, trainy, 1e9);

        int errorQda = 0;
        for (int i = 0; i < testx.length; i++)
            if (qda.predict(testx[i]) != testy[i])
                errorQda++;

        System.out.format("QDA Error rate = %.2f%%\n", 100.0 * errorQda / testx.length);
    }

    private static void runDt(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        DecisionTree dt = new DecisionTree(trainx, trainy, 100);

        int errorDt = 0;
        for (int i = 0; i < testx.length; i++)
            if (dt.predict(testx[i]) != testy[i])
                errorDt++;

        System.out.format("DT Error rate = %.2f%%\n", 100.0 * errorDt / testx.length);
    }

    private static void runRf(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        RandomForest rf = new RandomForest(trainx, trainy, 100);

        int errorRf = 0;
        for (int i = 0; i < testx.length; i++)
            if (rf.predict(testx[i]) != testy[i])
                errorRf++;

        System.out.format("RF Error rate = %.2f%%\n", 100.0 * errorRf / testx.length);
    }

    private static void runMlp(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        NeuralNetwork nn = new NeuralNetwork(ErrorFunction.LEAST_MEAN_SQUARES, trainx[0].length, trainx[0].length / 2, trainy.length);

        nn.setLearningRate(0.01);
        nn.setMomentum(0.5);
        nn.learn(trainx, trainy);
        nn.setWeightDecay(0);

        int errorNn = 0;
        for (int i = 0; i < testx.length; i++)
            if (nn.predict(testx[i]) != testy[i])
                errorNn++;

        System.out.format("NN Error rate = %.2f%%\n", 100.0 * errorNn / testx.length);
    }

    private static void runRbf(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        double[][] centers = new double[trainx.length][trainx[0].length];
        for (int i = 0; i < trainx.length; i++)
            for (int j = 0; j < trainx[0].length; j++)
                centers[i][j] = Math.random();

        RBFNetwork<double[]> rbf = new RBFNetwork<>(trainx, trainy, new EuclideanDistance(), new GaussianRadialBasis(), centers);

        int errorRf = 0;
        for (int i = 0; i < testx.length; i++)
            if (rbf.predict(testx[i]) != testy[i])
                errorRf++;

        System.out.format("RBF Error rate = %.2f%%\n", 100.0 * errorRf / testx.length);
    }

    private static void runSvm(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        SVM<double[]> svm = new SVM<>(new GaussianKernel(8.0), 5.0, smile.math.Math.max(trainy) + 1);

        svm.learn(trainx, trainy);
        svm.finish();

        int errorSvm = 0;
        for (int i = 0; i < testx.length; i++)
            if (svm.predict(testx[i]) != testy[i])
                errorSvm++;

        System.out.format("SVM Error rate = %.2f%%\n", 100.0 * errorSvm / testx.length);
    }

    private static void runKnn(double[][] trainx, int[] trainy, double[][] testx, int[] testy)
    {
        KNN<double[]> knn = new KNN<>(trainx, trainy, new EuclideanDistance());

        int errorKnn = 0;
        for (int i = 0; i < testx.length; i++)
            if (knn.predict(testx[i]) != testy[i])
                errorKnn++;

        System.out.format("kNN Error rate = %.2f%%\n", 100.0 * errorKnn / testx.length);
    }

    private static void plot(double[][] x, int[] y)
    {
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setBounds(100, 100, 768, 576);
        JPanel contentPane = new JPanel();
        contentPane.setLayout(new BorderLayout());
        frame.setContentPane(contentPane);

        PlotCanvas plot = ScatterPlot.plot(x, y, 'o', new Color[]
        { Color.BLUE, Color.RED });
        contentPane.add(plot, BorderLayout.CENTER);

        frame.setVisible(true);
    }
}
