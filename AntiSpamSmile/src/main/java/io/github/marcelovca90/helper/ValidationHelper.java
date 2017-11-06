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

import smile.classification.Classifier;
import smile.validation.Accuracy;
import smile.validation.FMeasure;
import smile.validation.Sensitivity;
import smile.validation.Specificity;

public class ValidationHelper
{
    public static void validate(Classifier<double[]> classifier, double[][] x, int[] y)
    {
        int[] truth = y;
        int[] prediction = new int[truth.length];
        for (int i = 0; i < x.length; i++)
            prediction[i] = classifier.predict(x[i]);

        double accuracy = 100.0 * new Accuracy().measure(truth, prediction);
        double sensitivity = 100.0 * new Sensitivity().measure(truth, prediction);
        double specificity = 100.0 * new Specificity().measure(truth, prediction);
        double fmeasure = 100.0 * new FMeasure().measure(truth, prediction);

        assert (!Double.isNaN(accuracy)) : "accuracy must be a double";
        assert (!Double.isNaN(sensitivity)) : "sensitivity must be a double";
        assert (!Double.isNaN(specificity)) : "specificity must be a double";
        assert (!Double.isNaN(fmeasure)) : "fmeasure must be a double";

        System.out.println(String.format("Accuracy: %.2f", accuracy));
        System.out.println(String.format("Sensitivity: %.2f", sensitivity));
        System.out.println(String.format("Specificity: %.2f", specificity));
        System.out.println(String.format("FMeasure: %.2f", fmeasure));
    }
}
