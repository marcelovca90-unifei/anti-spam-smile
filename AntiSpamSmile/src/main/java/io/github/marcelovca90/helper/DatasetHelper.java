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
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.arturmkrtchyan.sizeof4j.SizeOf;

import io.github.marcelovca90.common.ClassType;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;

public class DatasetHelper
{
    private static final Logger LOGGER = LoggerFactory.getLogger(DatasetHelper.class);
    private static final int SIZE_INT = SizeOf.intSize();
    private static final int SIZE_DOUBLE = SizeOf.doubleSize();

    public static Set<Triple<String, Integer, Integer>> loadMetadata(String filename) throws IOException
    {
        Set<Triple<String, Integer, Integer>> metadata = new LinkedHashSet<>();

        Files.readAllLines(Paths.get(filename)).stream().filter(line -> !StringUtils.isEmpty(line) && !line.startsWith("#")).forEach(line -> {
            // replaces the user home symbol (~) with the actual folder path
            line = line.replace("~", System.getProperty("user.home"));
            String[] parts = line.split(",");
            String folder = parts[0];
            Integer emptyHamAmount = Integer.parseInt(parts[1]);
            Integer emptySpamAmount = Integer.parseInt(parts[2]);

            // add triple to metadata set
            metadata.add(Triple.of(folder, emptyHamAmount, emptySpamAmount));
        });

        return metadata;
    }

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
        for (int i = 0; i < numberOfAttributes; i++)
            atts[i] = new NumericAttribute("x" + i);
        Attribute response = new NominalAttribute("y");

        // create data set
        AttributeDataset dataset = new AttributeDataset("dataset", atts, response);

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
                dataset.add(x, classType.ordinal());
                doubleBuffer.clear();
            }
        }
        inputStream.close();

        assert dataset.size() == numberOfInstances;
        assert dataset.get(0).x.length == numberOfAttributes;

        return dataset;
    }

    public static AttributeDataset mergeDataSets(AttributeDataset... datasets)
    {
        AtomicInteger totalLength = new AtomicInteger(0);
        AttributeDataset mergedSet = new AttributeDataset("mergedDataSet", datasets[0].attributes(), datasets[0].response());
        Arrays.stream(datasets).forEach(dataset -> {
            totalLength.set(totalLength.get() + dataset.size());
            dataset.forEach(mergedSet::add);
        });

        assert mergedSet.size() == totalLength.get();

        return mergedSet;
    }

    public static AttributeDataset shuffle(AttributeDataset dataset, int seed)
    {
        double[][] x = dataset.toArray(new double[dataset.size()][]);
        int[] y = dataset.toArray(new int[dataset.size()]);

        Random random = new Random(seed);
        AttributeDataset shuffledDataSet = new AttributeDataset("shuffledDataSet", dataset.attributes(), dataset.response());

        for (int i = 0; i < dataset.size(); i++)
        {
            int j = random.nextInt(dataset.size());

            double[] tempX = x[i];
            x[i] = x[j];
            x[j] = tempX;

            int tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;

            shuffledDataSet.add(x[i], y[i]);
        }

        assert shuffledDataSet.size() == dataset.size();

        return shuffledDataSet;
    }

    public static Pair<AttributeDataset, AttributeDataset> split(AttributeDataset dataset, double splitPercent)
    {
        AttributeDataset trainSet = new AttributeDataset("trainSet", dataset.attributes(), dataset.response());
        for (int i = 0; i < (int) (splitPercent * dataset.size()); i++)
            trainSet.add(dataset.get(i));

        AttributeDataset testSet = new AttributeDataset("testSet", dataset.attributes(), dataset.response());
        for (int i = (int) (splitPercent * dataset.size()); i < dataset.size(); i++)
            testSet.add(dataset.get(i));

        assert trainSet.size() + testSet.size() == dataset.size();

        return Pair.of(trainSet, testSet);
    }
}
