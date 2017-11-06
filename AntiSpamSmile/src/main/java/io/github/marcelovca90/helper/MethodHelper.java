//**********************************************************************
// Copyright (c) 2017 Telefonaktiebolaget LM Ericsson, Sweden.
// All rights reserved.
// The Copyright to the computer program(s) herein is the property of
// Telefonaktiebolaget LM Ericsson, Sweden.
// The program(s) may be used and/or copied with the written permission
// from Telefonaktiebolaget LM Ericsson or in accordance with the terms
// and conditions stipulated in the agreement/contract under which the
// program(s) have been supplied.
// **********************************************************************
package io.github.marcelovca90.helper;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import smile.classification.AdaBoost;
import smile.classification.Classifier;
import smile.classification.DecisionTree;
import smile.classification.FLD;
import smile.classification.GradientTreeBoost;
import smile.classification.KNN;
import smile.classification.LDA;
import smile.classification.LogisticRegression;
import smile.classification.Maxent;
import smile.classification.NeuralNetwork;
import smile.classification.NeuralNetwork.ErrorFunction;
import smile.classification.QDA;
import smile.classification.RBFNetwork;
import smile.classification.RDA;
import smile.classification.RandomForest;
import smile.classification.SVM;
import smile.math.distance.EuclideanDistance;
import smile.math.kernel.GaussianKernel;
import smile.math.rbf.RadialBasisFunction;
import smile.util.SmileUtils;

@SuppressWarnings ("rawtypes")
public class MethodHelper
{
    private static final Logger LOGGER = LoggerFactory.getLogger(MethodHelper.class);
    private static final Map<Class<? extends Classifier>, Callable<?>> METHODS = new HashMap<>();

    public static void init(double[][] x, int[] y)
    {
        // K-Nearest Neighbor
        METHODS.put(KNN.class, () -> new KNN<>(x, y, new EuclideanDistance()));

        // Linear Discriminant Analysis
        METHODS.put(LDA.class, () -> new LDA(x, y));

        // Fisher's Linear Discriminant
        METHODS.put(FLD.class, () -> new FLD(x, y));

        // Quadratic Discriminant analysis
        METHODS.put(QDA.class, () -> new QDA(x, y));

        // Regularized Discriminant Analysis
        METHODS.put(RDA.class, () -> new RDA(x, y, 0.5));

        // Logistic Regression
        METHODS.put(LogisticRegression.class, () -> new LogisticRegression(x, y));

        // Maximum Entropy Classifier
        METHODS.put(Maxent.class, () ->
        {
            int[][] binx = new int[x.length][x[0].length];
            for (int i = 0; i < x.length; i++)
                for (int j = 0; j < x[0].length; j++)
                    binx[i][j] = Math.abs(x[i][j]) < 1E-18 ? 0 : 1;
            return new Maxent(x[0].length, binx, y);
        });

        // Multilayer Perceptron Neural Network
        METHODS.put(NeuralNetwork.class, () ->
        {
            int i = x[0].length, o = 1, k = x.length;
            int h = (int) ((-1 * (i + o)) + Math.sqrt(Math.pow(i + o, 2) + (4 * k)) / 2);
            return new NeuralNetwork(ErrorFunction.LEAST_MEAN_SQUARES, x[0].length, h, h, 1);
        });

        // Radial Basis Function Networks
        METHODS.put(RBFNetwork.class, () ->
        {
            double[][] centers = new double[10][];
            RadialBasisFunction[] basis = SmileUtils.learnGaussianRadialBasis(x, centers, 5.0);
            return new RBFNetwork<>(x, y, new EuclideanDistance(), basis, centers);
        });

        // Support Vector Machines
        METHODS.put(SVM.class, () -> new SVM<>(new GaussianKernel(8.0), 5.0, smile.math.Math.max(y) + 1));

        // Decision Trees
        METHODS.put(DecisionTree.class, () -> new DecisionTree(x, y, 100));

        // Random Forest
        METHODS.put(RandomForest.class, () -> new RandomForest(x, y, 100));

        // Gradient Boosted Trees
        METHODS.put(GradientTreeBoost.class, () -> new GradientTreeBoost(x, y, 100));

        // AdaBoost
        METHODS.put(AdaBoost.class, () -> new AdaBoost(x, y, 100));
    }

    public static Classifier forClass(Class<? extends Classifier> clazz) throws Exception
    {
        return clazz.cast(METHODS.get(clazz).call());
    }
}
