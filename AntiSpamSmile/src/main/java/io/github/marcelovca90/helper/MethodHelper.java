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

@SuppressWarnings ("rawtypes")
public class MethodHelper
{
    private static final Map<Class<? extends Classifier>, Callable<?>> methods = new HashMap<>();

    public static void init(double[][] x, int[] y)
    {
        // K-Nearest Neighbor
        methods.put(KNN.class, () -> new KNN<>(x, y, new EuclideanDistance()));

        // Linear Discriminant Analysis
        methods.put(LDA.class, () -> new LDA(x, y));

        // Fisher's Linear Discriminant
        methods.put(FLD.class, () -> new FLD(x, y));

        // Quadratic Discriminant analysis
        methods.put(QDA.class, () -> new QDA(x, y));

        // Regularized Discriminant Analysis
        methods.put(RDA.class, () -> new RDA(x, y, 0.5));

        // Logistic Regression
        methods.put(LogisticRegression.class, () -> new LogisticRegression(x, y));

        // Maximum Entropy Classifier
        methods.put(Maxent.class, null); // TODO implement runnable for constructor

        // Multilayer Perceptron Neural Network
        methods.put(NeuralNetwork.class, () -> new NeuralNetwork(ErrorFunction.LEAST_MEAN_SQUARES, x[0].length, x[0].length / 2, 1));

        // Radial Basis Function Networks
        methods.put(RBFNetwork.class, null); // TODO implement runnable for constructor

        // Support Vector Machines
        methods.put(SVM.class, () -> new SVM<>(new GaussianKernel(8.0), 5.0, smile.math.Math.max(y) + 1));

        // Decision Trees
        methods.put(DecisionTree.class, () -> new DecisionTree(x, y, 100));

        // Random Forest
        methods.put(RandomForest.class, () -> new RandomForest(x, y, 100));

        // Gradient Boosted Trees
        methods.put(GradientTreeBoost.class, () -> new GradientTreeBoost(x, y, 100));

        // AdaBoost
        methods.put(AdaBoost.class, () -> new AdaBoost(x, y, 100));
    }

    public static Classifier forClass(Class<? extends Classifier> clazz) throws Exception
    {
        return clazz.cast(methods.get(clazz).call());
    }
}
