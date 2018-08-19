/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package il.ac.openu.bestparents.core;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.TreeMap;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.ContingencyTables;
import weka.core.Instances;

/**
 * @author Andrew Kreimer - algonell.com
 */
public class BestParentsAndChildrenRecursiveSearch extends SearchAlgorithm {

	private static final long serialVersionUID = 2467629575499347683L;
	
	private int m_nMaxNrOfChildren;

	/**
	 * @param bayesNet
	 *            the network
	 * @param instances
	 *            the data to work with
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public void search(BayesNet bayesNet, Instances instances) throws Exception {
		// contingency table for each attribute X attribute matrix
		double attributeMatrix[][][][] = new double[instances.numAttributes()][instances.numAttributes()][][];

		// allocate
		for (int j = 0; j < instances.numAttributes(); j++) {
			for (int k = 0; k < j; k++) {
				if (j == k)
					continue;
				attributeMatrix[j][k] = new double[instances.attribute(j).numValues()][instances.attribute(k).numValues()];
			}
		}

		// count instantiations
		for (int n = 0; n < instances.numInstances(); n++) {
			for (int i = 0; i < instances.numAttributes(); i++) {
				for (int j = 0; j < i; j++) {
					int iAttrIndex = (int) instances.instance(n).value(i);
					int jAttrIndex = (int) instances.instance(n).value(j);
					attributeMatrix[i][j][iAttrIndex][jAttrIndex]++;
				}
			}
		}

		// for each attribute with index i: map<entropy, parent index>, keeping
		// the map sorted
		ArrayList<TreeMap<Double, Integer>> attributeBestParentsList = new ArrayList<TreeMap<Double, Integer>>();
		ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList = new ArrayList<TreeMap<Double, Integer>>();
		ArrayList<TreeMap<Double, Integer>> attributeBestParentsAndChildrenList = new ArrayList<TreeMap<Double, Integer>>();

		// allocate
		for (int i = 0; i < instances.numAttributes(); i++) {
			TreeMap<Double, Integer> tmpTreeMap = null;
			tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestParentsList.add(i, tmpTreeMap);
			tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestChildrenList.add(i, tmpTreeMap);
			tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestParentsAndChildrenList.add(i, tmpTreeMap);
		}

		// map<entropy, rule(string)>
		TreeMap<Double, String> entropyRuleMap = new TreeMap<Double, String>();

		// map<entropy, rule(attributeChildIndex <- attributeParentIndex)>
		TreeMap<Double, Entry<Integer, Integer>> entropyChildFromParentMap = new TreeMap<Double, Entry<Integer, Integer>>();

		// map<entropy, rule(attributeParentIndex -> attributeChildIndex)>
		TreeMap<Double, Entry<Integer, Integer>> entropyParentToChildMap = new TreeMap<Double, Entry<Integer, Integer>>();

		// calculate conditional entropy for contingency tables
		for (int i = 0; i < instances.numAttributes(); i++) {
			for (int j = 0; j < i; j++) {
				double entropyConditionedOnRows = ContingencyTables.entropyConditionedOnRows(attributeMatrix[i][j]);
				double entropyConditionedOnColumns = ContingencyTables.entropyConditionedOnColumns(attributeMatrix[i][j]);

				double lowestEntropy = (entropyConditionedOnRows < entropyConditionedOnColumns) ? entropyConditionedOnRows : entropyConditionedOnColumns;

				// save current rule
				String arc = (entropyConditionedOnRows < entropyConditionedOnColumns) ? instances.attribute(j).name() + " <- " + instances.attribute(i).name() : instances.attribute(i).name() + " <- " + instances.attribute(j).name();
				entropyRuleMap.put(lowestEntropy, arc);

				if (entropyConditionedOnRows < entropyConditionedOnColumns) {
					attributeBestParentsList.get(j).put(lowestEntropy, i);
					attributeBestChildrenList.get(i).put(lowestEntropy, j);
					entropyChildFromParentMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer, Integer>(j, i));
					entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer, Integer>(i, j));
				} else {
					attributeBestParentsList.get(i).put(lowestEntropy, j);
					attributeBestChildrenList.get(j).put(lowestEntropy, i);
					entropyChildFromParentMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer, Integer>(i, j));
					entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer, Integer>(j, i));
				}
			}
		}

		addBestParentsAndChildrenIterative(bayesNet, instances, instances.numAttributes() - 1, attributeBestChildrenList, attributeBestParentsList);

		// Start from class variable
		// boolean blackList[] = new boolean[instances.numAttributes()];
		// blackList[instances.numAttributes() - 1] = true;
		// addBestParentsAndChildren(bayesNet, instances,
		// attributeBestChildrenList, attributeBestParentsList,
		// instances.numAttributes() - 1, blackList);
		//
		// //use the remaining attributes
		// System.out.println("\nBestParentsAndChildrenRecursiveSearch Use
		// remaining Attributes...\n");
		// for (int i = 0; i < blackList.length; i++) {
		// if(!blackList[i]){
		// addBestParentsAndChildren(bayesNet, instances,
		// attributeBestChildrenList, attributeBestParentsList, i, blackList);
		// }
		// }

//		System.out.println("\nBestParentsAndChildrenRecursiveSearch: complete");
	} // buildStructure

	/**
	 * Sets the max number of parents
	 * 
	 * @param nMaxNrOfParents
	 *            the max number of parents
	 */
	public void setMaxNrOfParents(int nMaxNrOfParents) {
		m_nMaxNrOfParents = nMaxNrOfParents;
	}

	/**
	 * Gets the max number of parents.
	 * 
	 * @return the max number of parents
	 */
	public int getMaxNrOfParents() {
		return m_nMaxNrOfParents;
	}

	/**
	 * Sets the max number of children
	 * 
	 * @param nMaxNrOfChildren
	 *            the max number of children
	 */
	public void setMaxNrOfChildren(int nMaxNrOfChildren) {
		m_nMaxNrOfChildren = nMaxNrOfChildren;
	}

	/**
	 * Gets the max number of children.
	 * 
	 * @return the max number of children
	 */
	public int getMaxNrOfChildren() {
		return m_nMaxNrOfChildren;
	}

	/**
	 * Recursive algorithm: start with class by adding best parent and child,
	 * expand for child and parent the same algorithm till no more attributes
	 * left
	 * 
	 * @param bayesNet
	 * @param instances
	 * @param attributeBestChildrenList
	 *            - list of maps
	 * @param attributeBestParentsList
	 *            - list of maps
	 * @param i
	 *            - attribute
	 */
	@SuppressWarnings("unused")
	private static void addBestParentsAndChildren(BayesNet bayesNet, Instances instances, ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList, ArrayList<TreeMap<Double, Integer>> attributeBestParentsList, int i, boolean blackList[]) {
		TreeMap<Double, Integer> tmpBestChildrenMap = attributeBestChildrenList.get(i);
		TreeMap<Double, Integer> tmpBestParentsMap = attributeBestParentsList.get(i);

		double bestChildKey = Double.MAX_VALUE;// +infinity
		double bestParentKey = Double.MAX_VALUE;// +infinity

		if (tmpBestChildrenMap.keySet().toArray().length != 0) {
			bestChildKey = (double) tmpBestChildrenMap.keySet().toArray()[0];
		}

		if (tmpBestParentsMap.keySet().toArray().length != 0) {
			bestParentKey = (double) tmpBestParentsMap.keySet().toArray()[0];
		}

		Integer bestChild = tmpBestChildrenMap.get(bestChildKey);
		Integer bestParent = tmpBestParentsMap.get(bestParentKey);

//		System.out.println("BestParentsAndChildrenRecursiveSearch.addBestParentsAndChildren() Attribute: " + i);
//		System.out.println("BestParentsAndChildrenRecursiveSearch.addBestParentsAndChildren() tmpBestChildrenMap.get(bestChildKey): " + bestChild);
//		System.out.println("BestParentsAndChildrenRecursiveSearch.addBestParentsAndChildren() tmpBestParentsMap.get(bestParentKey): " + bestParent);
//		System.out.println("BestParentsAndChildrenRecursiveSearch.addBestParentsAndChildren() " + i + ": " + bestChildKey + "(" + bestChild + ") vs. " + bestParentKey + "(" + bestParent + ")");

		boolean expandToChild = false;
		boolean expandToParent = false;

		// order matters, best child or parent could be stolen because of
		// execution order!
		if (bestChildKey < bestParentKey) {
			if (bestChild != null && !blackList[bestChild]) {
				bayesNet.getParentSet(bestChild).addParent(i, instances);
				blackList[bestChild] = true;
				expandToChild = true;
				tmpBestChildrenMap.remove(bestChildKey);
			}

			if (bestParent != null && !blackList[bestParent]) {
				bayesNet.getParentSet(i).addParent(bestParent, instances);
				blackList[bestParent] = true;
				expandToParent = true;
				tmpBestParentsMap.remove(bestParentKey);
			}

			if (expandToChild)
				addBestParentsAndChildren(bayesNet, instances, attributeBestChildrenList, attributeBestParentsList, bestChild, blackList);
			if (expandToParent)
				addBestParentsAndChildren(bayesNet, instances, attributeBestChildrenList, attributeBestParentsList, bestParent, blackList);
		} else {
			if (bestParent != null && !blackList[bestParent]) {
				bayesNet.getParentSet(i).addParent(bestParent, instances);
				blackList[bestParent] = true;
				expandToParent = true;
				tmpBestParentsMap.remove(bestParentKey);
			}

			if (bestChild != null && !blackList[bestChild]) {
				bayesNet.getParentSet(bestChild).addParent(i, instances);
				blackList[bestChild] = true;
				expandToChild = true;
				tmpBestChildrenMap.remove(bestChildKey);
			}

			if (expandToParent)
				addBestParentsAndChildren(bayesNet, instances, attributeBestChildrenList, attributeBestParentsList, bestParent, blackList);
			if (expandToChild)
				addBestParentsAndChildren(bayesNet, instances, attributeBestChildrenList, attributeBestParentsList, bestChild, blackList);
		}
	}

	/**
	 * Iterative algorithm: start with class by adding best parent and child,
	 * expand for child and parent the same algorithm till no more attributes
	 * left. Use queue instead of recursive expansion.
	 * 
	 * @param bayesNet
	 * @param instances
	 * @param startingAttribute
	 * @param bestChildrenList
	 * @param bestParentsList
	 */
	public void addBestParentsAndChildrenIterative(BayesNet bayesNet, Instances instances, int startingAttribute, ArrayList<TreeMap<Double, Integer>> bestChildrenList, ArrayList<TreeMap<Double, Integer>> bestParentsList) {
		// keeping the expansion order is important, expand to the lowest
		// conditional entropies first
		TreeMap<Double, Integer> expansionOrder = new TreeMap<Double, Integer>();

		// blackList: mark Class as used
		boolean blackList[] = new boolean[instances.numAttributes()];
		blackList[startingAttribute] = true;

		// expansion queue
		Queue<Integer> queue = new LinkedList<Integer>();

		// start with class
		queue.add(startingAttribute);

		// while there are attributes to expand
		while (!queue.isEmpty()) {
//			System.out.println("Start - Current: " + queue + " Next: " + expansionOrder.values());

			// current attribute
			Integer attribute = queue.remove();
			blackList[attribute] = true;

			// add best parents
			for (int i = 0; i < getMaxNrOfParents(); i++) {
				Object arr[] = bestParentsList.get(attribute).keySet().toArray();

				// if there are rules
				if (arr.length == 0) {
					break;
				}

				Double key = (Double) arr[i];
				Integer val = bestParentsList.get(attribute).get(key);

				if (i < arr.length && !blackList[val] 
						/*&& bayesNet.getParentSet(attribute).getNrOfParents() < getMaxNrOfParents()*/) {
					bayesNet.getParentSet(attribute).addParent(val, instances);
					expansionOrder.put(key, val);
					blackList[val] = true;
				}
			}

			// add best children
			for (int i = 0; i < getMaxNrOfChildren(); i++) {
				Object arr[] = bestChildrenList.get(attribute).keySet().toArray();

				// if there are rules
				if (arr.length == 0) {
					break;
				}

				Double key = (Double) arr[i];
				Integer val = bestChildrenList.get(attribute).get(key);

				if (i < arr.length && !blackList[val] 
						/*&& bayesNet.getParentSet(val).getNrOfParents() < getMaxNrOfParents()*/) {
					bayesNet.getParentSet(val).addParent(i, instances);
					expansionOrder.put(key, val);
					blackList[val] = true;
				}
			}

//			System.out.println("End - Current: " + queue + " Next: " + expansionOrder.values() + "\n");

			//if queue is empty, proceed with expansion by order 
			if (queue.isEmpty()) {
				queue = new LinkedList<Integer>();
				
				for (Double key : expansionOrder.keySet()) {
					queue.add(expansionOrder.get(key));
				}
				
				expansionOrder = new TreeMap<Double, Integer>();
			}
		}
	}
}