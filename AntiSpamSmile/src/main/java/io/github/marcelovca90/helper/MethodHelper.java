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

@SuppressWarnings("rawtypes")
public class MethodHelper
{
    private static final Logger LOGGER = LoggerFactory.getLogger(MethodHelper.class);
    private static final Map<Class<? extends Classifier>, Callable<?>> METHODS = new HashMap<>();

    public static void init(double[][] x, int[] y)
    {
        // K-Nearest Neighbor
        METHODS.put(KNN.class, () -> new KNN.Trainer<>(new EuclideanDistance(), 10).train(x, y));

        // Linear Discriminant Analysis
        METHODS.put(LDA.class, () -> new LDA.Trainer().train(x, y));

        // Fisher's Linear Discriminant
        METHODS.put(FLD.class, () -> new FLD.Trainer().train(x, y));

        // Quadratic Discriminant analysis
        METHODS.put(QDA.class, () -> new QDA.Trainer().train(x, y));

        // Regularized Discriminant Analysis
        METHODS.put(RDA.class, () -> new RDA.Trainer(1E-4).train(x, y));

        // Logistic Regression
        METHODS.put(LogisticRegression.class, () -> new LogisticRegression.Trainer().train(x, y));

        // Maximum Entropy Classifier
        METHODS.put(Maxent.class, () ->
        {
            int[][] binx = new int[x.length][x[0].length];
            for (int i = 0; i < x.length; i++)
                for (int j = 0; j < x[0].length; j++)
                    binx[i][j] = Math.abs(x[i][j]) < 1E-18 ? 0 : 1;
            return new Maxent.Trainer(x[0].length).train(binx, y);
        });

        // Multilayer Perceptron Neural Network
        METHODS.put(NeuralNetwork.class, () -> new NeuralNetwork.Trainer(ErrorFunction.CROSS_ENTROPY, x[0].length, (x[0].length + 1) / 2, 1).train(x, y));

        // Radial Basis Function Networks
        METHODS.put(RBFNetwork.class, () -> new RBFNetwork.Trainer<>(new EuclideanDistance()).train(x, y));

        // Support Vector Machines
        METHODS.put(SVM.class, () -> new SVM.Trainer<>(new GaussianKernel(8.0), 5.0, smile.math.Math.max(y) + 1).train(x, y));

        // Decision Trees
        METHODS.put(DecisionTree.class, () -> new DecisionTree.Trainer(100).train(x, y));

        // Random Forest
        METHODS.put(RandomForest.class, () -> new RandomForest.Trainer(100).train(x, y));

        // Gradient Boosted Trees
        METHODS.put(GradientTreeBoost.class, () -> new GradientTreeBoost.Trainer(100).train(x, y));

        // AdaBoost
        METHODS.put(AdaBoost.class, () -> new AdaBoost.Trainer(100).train(x, y));
    }

    public static Classifier forClass(Class<? extends Classifier> clazz) throws Exception
    {
        return clazz.cast(METHODS.get(clazz).call());
    }
}
