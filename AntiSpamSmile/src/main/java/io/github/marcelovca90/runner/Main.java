package io.github.marcelovca90.runner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.primes.Primes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.marcelovca90.helper.DatasetHelper;
import io.github.marcelovca90.helper.FeatureSelectionHelper;
import io.github.marcelovca90.helper.MethodHelper;
import io.github.marcelovca90.helper.ValidationHelper;
import smile.classification.Classifier;
import smile.classification.DecisionTree;
import smile.classification.RandomForest;
import smile.data.AttributeDataset;

public class Main
{
    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);
    private static final String METADATA_PATH = "/Users/marcelocysneiros/git/anti-spam-weka-data/2017_BASE2/metadataUpTo32.txt";

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public static void main(String[] args) throws Exception
    {
        Class<? extends Classifier>[] clazzes = new Class[] {
                DecisionTree.class, RandomForest.class
        };

        for (Class clazz : clazzes)
        {
            for (Triple<String, Integer, Integer> metadatum : DatasetHelper.loadMetadata(METADATA_PATH))
            {
                // read data
                AttributeDataset dataset = DatasetHelper.load(metadatum);

                // select features
                int noFeaturesBefore = dataset.attributes().length;
                dataset = FeatureSelectionHelper.sumSquaresRatio(dataset);
                int noFeaturesAfter = dataset.attributes().length;

                // initialize RNG seed
                int seed = 2;

                LOGGER.info("{} with ({} -> {}) features", clazz.getName(), noFeaturesBefore, noFeaturesAfter);

                // perform 10 executions
                for (int run = 0; run < 10; run++)
                {
                    // shuffle data
                    dataset = DatasetHelper.shuffle(dataset, seed);

                    // build train/test data
                    Pair<AttributeDataset, AttributeDataset> pair = DatasetHelper.split(dataset, 0.5);
                    AttributeDataset train = pair.getLeft();
                    AttributeDataset test = pair.getRight();
                    double[][] xTrain = train.toArray(new double[train.size()][]);
                    int[] yTrain = train.toArray(new int[train.size()]);
                    double[][] xTest = test.toArray(new double[test.size()][]);
                    int[] yTest = test.toArray(new int[test.size()]);

                    // train and test classifier
                    MethodHelper.init(xTrain, yTrain);
                    Classifier classifier = MethodHelper.forClass(clazz);
                    ValidationHelper.aggregate(classifier, xTest, yTest);

                    // update RNG seed
                    seed = Primes.nextPrime(seed + 1);
                }

                // print consolidated statistics for this method
                ValidationHelper.consolidate(clazz);
            }
        }
    }
}
