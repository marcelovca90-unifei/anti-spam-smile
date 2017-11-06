package io.github.marcelovca90.runner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.primes.Primes;

import io.github.marcelovca90.common.ClassType;
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
    @SuppressWarnings (
    { "rawtypes", "unchecked" })
    public static void main(String[] args) throws Exception
    {
        Class<? extends Classifier>[] clazzes = new Class[]
        {
                DecisionTree.class, RandomForest.class
        };

        for (Class clazz : clazzes)
        {
            for (int features = 8; features <= 1024; features *= 2)
            {
                // TODO: load file names from metadata
                final String filenameHam = "C:\\Users\\marcelovca90\\git\\anti-spam-weka-data\\2017_BASE2\\2017_BASE2_SPAM_ASSASSIN\\MI\\" + features + "\\ham";
                final String filenameSpam = "C:\\Users\\marcelovca90\\git\\anti-spam-weka-data\\2017_BASE2\\2017_BASE2_SPAM_ASSASSIN\\MI\\" + features + "\\spam";

                // read data
                AttributeDataset ham = DatasetHelper.read(filenameHam, ClassType.HAM);
                AttributeDataset spam = DatasetHelper.read(filenameSpam, ClassType.SPAM);
                AttributeDataset dataset = DatasetHelper.mergeDataSets(ham, spam);

                // select features
                dataset = FeatureSelectionHelper.sumSquaresRatio(dataset);

                // initialize rng seed
                int seed = 2;

                System.out.println(clazz.getName() + " with " + features + " features");
                Thread.sleep(3000);

                // run 10 executions
                for (int run = 0; run < 10; run++)
                {
                    // shuffle data
                    dataset = DatasetHelper.shuffle(dataset, seed);

                    // build train/test data
                    Pair<AttributeDataset, AttributeDataset> pair = DatasetHelper.split(dataset, 0.5);
                    AttributeDataset train = pair.getLeft();
                    AttributeDataset test = pair.getRight();
                    double[][] trainx = train.toArray(new double[train.size()][]);
                    int[] trainy = train.toArray(new int[train.size()]);
                    double[][] testx = test.toArray(new double[test.size()][]);
                    int[] testy = test.toArray(new int[test.size()]);

                    // train and test classifier
                    MethodHelper.init(trainx, trainy);
                    Classifier classifier = MethodHelper.forClass(clazz);
                    ValidationHelper.aggregate(classifier, testx, testy);

                    // update rng seed
                    seed = Primes.nextPrime(seed + 1);
                }

                ValidationHelper.consolidate(clazz);

                System.out.println();
            }
        }
    }
}
