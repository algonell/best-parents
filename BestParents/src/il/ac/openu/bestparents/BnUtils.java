package il.ac.openu.bestparents;

import java.util.Enumeration;
import java.util.Map;

import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Holds misc utils for the new Bayesian Network structure learning algorithm.
 * 
 * @author Andrew Kreimer
 */
public abstract class BnUtils {
	
	/**
	 * Creates attribute distribution description
	 * 
	 * @param attribute
	 * @param distributions
	 * @return
	 */
	public static String makeAttributeDistributionsStr(Attribute attribute, double distributions[]) {
		StringBuilder sb = new StringBuilder();

		for (int i = 0; i < distributions.length - 1; i++) {
			sb.append(attribute.value(i) + " " + distributions[i] + "%, ");
		}

		sb.append(attribute.value(distributions.length - 1) + " " + distributions[distributions.length - 1] + "%");

		return sb.toString();
	}

	/**
	 * Returns number of children for given node
	 * 
	 * @param bayesNet
	 * @param iNode
	 * @return
	 */
	public static int countNumOfChildren(BayesNet bayesNet, Instances instances, int iNode) {
		int counter = 0;

		for (int i = 0; i < instances.numAttributes(); i++) {
			if (bayesNet.getParentSet(i).contains(iNode)) {
				counter++;
			}
		}

		return counter;
	}

	public static void printRulesMap(Map<Double, String> map) {
		for (Double key : map.keySet()) {
			System.out.println("key: " + key + " rule: " + map.get(key));
		}
	}

	public static void printParentSet(BayesNet bayesNet, Instances instances) {
		for (int i = 0; i < instances.numAttributes(); i++) {
			System.out.println("Attribute: " + i);
			System.out.println("CardinalityOfParents: " + bayesNet.getParentSet(i).getCardinalityOfParents());
			System.out.println("NrOfParents: " + bayesNet.getParentSet(i).getNrOfParents());
			for (int j = 0; j < bayesNet.getParentSet(i).getNrOfParents(); j++) {
				System.out.println("ParentSet[" + j + "]: " + bayesNet.getParentSet(i).getParent(j));
			}
		}
	}

	public static void printMatrix(double matrix[][]) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}

	/**
	 * Computes information gain for an attribute.
	 * 
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @return the information gain for the given attribute and data
	 * @throws Exception
	 *             if computation fails
	 */
	public static double computeInfoGain(Instances data, Attribute att) throws Exception {

		double infoGain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for (int j = 0; j < att.numValues(); j++) {
			if (splitData[j].numInstances() > 0) {
				infoGain -= ((double) splitData[j].numInstances() / (double) data.numInstances())
						* computeEntropy(splitData[j]);
			}
		}
		return infoGain;
	}

	/**
	 * Computes the entropy of a dataset.
	 * 
	 * @param data
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 * @throws Exception
	 *             if computation fails
	 */
	public static double computeEntropy(Instances data) throws Exception {

		double[] classCounts = new double[data.numClasses()];
		@SuppressWarnings("rawtypes")
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				entropy -= classCounts[j] * Utils.log2(classCounts[j]);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 * 
	 * @param data
	 *            the data which is to be split
	 * @param att
	 *            the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	public static Instances[] splitData(Instances data, Attribute att) {

		Instances[] splitData = new Instances[att.numValues()];
		for (int j = 0; j < att.numValues(); j++) {
			splitData[j] = new Instances(data, data.numInstances());
		}
		@SuppressWarnings("rawtypes")
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			splitData[(int) inst.value(att)].add(inst);
		}
		for (int i = 0; i < splitData.length; i++) {
			splitData[i].compactify();
		}
		return splitData;
	}
	
	private BnUtils() {}
	
}