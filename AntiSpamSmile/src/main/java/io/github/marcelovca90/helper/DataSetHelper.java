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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.tuple.Pair;

import com.arturmkrtchyan.sizeof4j.SizeOf;

import io.github.marcelovca90.common.ClassType;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;

public class DataSetHelper
{
    private static final int SIZE_INT = SizeOf.intSize();
    private static final int SIZE_DOUBLE = SizeOf.doubleSize();

    public static AttributeDataset read(String filename, ClassType classType) throws IOException
    {
        InputStream inputStream = new FileInputStream(filename);

        byte[] byteBufferA = new byte[SIZE_INT];
        inputStream.read(byteBufferA);
        int numberOfInstances = ByteBuffer.wrap(byteBufferA).getInt();

        byte[] byteBufferB = new byte[SIZE_INT];
        inputStream.read(byteBufferB);
        int numberOfAttributes = ByteBuffer.wrap(byteBufferB).getInt();

        // create attributes
        Attribute[] atts = new Attribute[numberOfAttributes];
        for (int i = 1; i <= numberOfAttributes; i++)
            atts[i - 1] = new NumericAttribute("x" + i);
        Attribute response = new NominalAttribute("y");

        // create data set
        AttributeDataset dataSet = new AttributeDataset("dataSet", atts, response);

        // create instance placeholder
        double[] x = new double[numberOfAttributes];

        byte[] byteBufferC = new byte[SIZE_DOUBLE];
        DoubleBuffer doubleBuffer = DoubleBuffer.allocate(numberOfAttributes);

        while (inputStream.read(byteBufferC) != -1)
        {
            doubleBuffer.put(ByteBuffer.wrap(byteBufferC).getDouble());
            if (!doubleBuffer.hasRemaining())
            {
                double[] values = doubleBuffer.array();
                for (int j = 0; j < numberOfAttributes; j++)
                    x[j] = values[j];
                dataSet.add(x, classType.ordinal());
                doubleBuffer.clear();
            }
        }
        inputStream.close();

        assert dataSet.size() == numberOfInstances;
        assert dataSet.get(0).x.length == numberOfAttributes;

        return dataSet;
    }

    public static AttributeDataset mergeDataSets(AttributeDataset... dataSets)
    {
        AtomicInteger totalLength = new AtomicInteger(0);
        AttributeDataset mergedSet = new AttributeDataset("mergedDataSet", dataSets[0].attributes(), dataSets[0].response());
        Arrays.stream(dataSets).forEach(dataSet -> {
            totalLength.set(totalLength.get() + dataSet.size());
            dataSet.forEach(mergedSet::add);
        });

        assert mergedSet.size() == totalLength.get();

        return mergedSet;
    }

    public static AttributeDataset shuffle(AttributeDataset dataSet, int seed)
    {
        double[][] x = dataSet.toArray(new double[dataSet.size()][]);
        int[] y = dataSet.toArray(new int[dataSet.size()]);

        Random random = new Random(seed);
        AttributeDataset shuffledDataSet = new AttributeDataset("shuffledDataSet", dataSet.attributes(), dataSet.response());

        for (int i = 0; i < dataSet.size(); i++)
        {
            int j = random.nextInt(dataSet.size());

            double[] tempX = x[i];
            x[i] = x[j];
            x[j] = tempX;

            int tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;

            shuffledDataSet.add(x[i], y[i]);
        }

        assert shuffledDataSet.size() == dataSet.size();

        return shuffledDataSet;
    }

    public static Pair<AttributeDataset, AttributeDataset> split(AttributeDataset dataSet, double splitPercent)
    {
        AttributeDataset trainSet = new AttributeDataset("trainSet", dataSet.attributes(), dataSet.response());
        for (int i = 0; i < (int) (splitPercent * dataSet.size()); i++)
            trainSet.add(dataSet.get(i));

        AttributeDataset testSet = new AttributeDataset("testSet", dataSet.attributes(), dataSet.response());
        for (int i = (int) (splitPercent * dataSet.size()); i < dataSet.size(); i++)
            testSet.add(dataSet.get(i));

        assert trainSet.size() + testSet.size() == dataSet.size();

        return Pair.of(trainSet, testSet);
    }
}
