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
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import smile.classification.Classifier;
import smile.validation.Accuracy;
import smile.validation.ClassificationMeasure;
import smile.validation.FMeasure;
import smile.validation.Sensitivity;
import smile.validation.Specificity;

@SuppressWarnings ("rawtypes")
public class ValidationHelper
{
    private static final Map<Class<? extends Classifier>, Map<Class<? extends ClassificationMeasure>, DescriptiveStatistics>> results = new HashMap<>();

    public static void aggregate(Classifier<double[]> classifier, double[][] x, int[] y)
    {
        results.putIfAbsent(classifier.getClass(), new LinkedHashMap<>());

        int[] truth = y;
        int[] prediction = new int[truth.length];
        for (int i = 0; i < x.length; i++)
            prediction[i] = classifier.predict(x[i]);

        double accuracy = 100.0 * new Accuracy().measure(truth, prediction);
        results.get(classifier.getClass()).putIfAbsent(Accuracy.class, new DescriptiveStatistics());
        results.get(classifier.getClass()).get(Accuracy.class).addValue(accuracy);
        assert !Double.isNaN(accuracy) : "accuracy must be a double";

        double sensitivity = 100.0 * new Sensitivity().measure(truth, prediction);
        results.get(classifier.getClass()).putIfAbsent(Sensitivity.class, new DescriptiveStatistics());
        results.get(classifier.getClass()).get(Sensitivity.class).addValue(sensitivity);
        assert !Double.isNaN(sensitivity) : "sensitivity must be a double";

        double specificity = 100.0 * new Specificity().measure(truth, prediction);
        results.get(classifier.getClass()).putIfAbsent(Specificity.class, new DescriptiveStatistics());
        results.get(classifier.getClass()).get(Specificity.class).addValue(specificity);
        assert !Double.isNaN(specificity) : "specificity must be a double";

        double fmeasure = 100.0 * new FMeasure().measure(truth, prediction);
        results.get(classifier.getClass()).putIfAbsent(FMeasure.class, new DescriptiveStatistics());
        results.get(classifier.getClass()).get(FMeasure.class).addValue(fmeasure);
        assert !Double.isNaN(fmeasure) : "fmeasure must be a double";
    }

    public static void consolidate(Class<? extends Classifier> clazz)
    {
        System.out.println(results.get(clazz).keySet().stream().map(k -> StringUtils.rightPad(k.getSimpleName(), 15)).collect(Collectors.joining("\t")));
        System.out.println(results.get(clazz).values().stream().map(v -> String.format("%.2f ± %.2f", v.getMean(), computeConfidenceInterval(v, 0.05))).collect(Collectors.joining("\t")));
    }

    private static double computeConfidenceInterval(DescriptiveStatistics statistics, double significance)
    {
        TDistribution tDist = new TDistribution(statistics.getN() - 1);
        double a = tDist.inverseCumulativeProbability(1.0 - significance / 2);
        return a * statistics.getStandardDeviation() / Math.sqrt(statistics.getN());
    }
}
