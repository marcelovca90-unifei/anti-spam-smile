package io.github.marcelovca90.runner;

import org.apache.commons.lang3.tuple.Pair;

import io.github.marcelovca90.common.ClassType;
import io.github.marcelovca90.helper.DatasetHelper;
import io.github.marcelovca90.helper.FeatureSelectionHelper;
import io.github.marcelovca90.helper.MethodHelper;
import io.github.marcelovca90.helper.ValidationHelper;
import smile.classification.Classifier;
import smile.classification.NeuralNetwork;
import smile.data.AttributeDataset;

public class Main
{
    @SuppressWarnings (
    { "rawtypes", "unchecked" })
    public static void main(String[] args) throws Exception
    {
        final String filenameHam = "C:\\Users\\marcelovca90\\git\\anti-spam-weka-data\\2017_BASE2\\2017_BASE2_SPAM_ASSASSIN\\MI\\16\\ham";
        final String filenameSpam = "C:\\Users\\marcelovca90\\git\\anti-spam-weka-data\\2017_BASE2\\2017_BASE2_SPAM_ASSASSIN\\MI\\16\\spam";

        // read data
        AttributeDataset ham = DatasetHelper.read(filenameHam, ClassType.HAM);
        AttributeDataset spam = DatasetHelper.read(filenameSpam, ClassType.SPAM);
        AttributeDataset dataset = DatasetHelper.mergeDataSets(ham, spam);

        // select features
        dataset = FeatureSelectionHelper.sumSquaresRatio(dataset);

        // shuffle data
        dataset = DatasetHelper.shuffle(dataset, 0);

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
        Classifier classifier = MethodHelper.forClass(NeuralNetwork.class);
        ValidationHelper.validate(classifier, testx, testy);
    }
}
