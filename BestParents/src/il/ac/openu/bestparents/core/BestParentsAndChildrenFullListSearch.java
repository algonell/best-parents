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
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.ContingencyTables;
import weka.core.Instances;

/**
 * @author Andrew Kreimer - algonell.com
 */
public class BestParentsAndChildrenFullListSearch extends SearchAlgorithm{
	
	private static final long serialVersionUID = -6875216741076169820L;
	
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
		double attributeMatrix[][][][] = new double[instances.numAttributes()][instances
				.numAttributes()][][];

		// allocate
		for (int j = 0; j < instances.numAttributes(); j++) {
			for (int k = 0; k < j; k++) {
				if (j == k)
					continue;
				attributeMatrix[j][k] = new double[instances.attribute(j)
						.numValues()][instances.attribute(k).numValues()];
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

		// for each attribute with index i: map<entropy, parent index>, keeping the map sorted
		ArrayList<TreeMap<Double, Integer>> attributeBestParentsList = new ArrayList<TreeMap<Double, Integer>>();
		ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList = new ArrayList<TreeMap<Double, Integer>>();
		ArrayList<TreeMap<Double, Integer>> attributeBestParentsAndChildrenList = new ArrayList<TreeMap<Double, Integer>>();
		
		//allocate
		for (int i = 0; i < instances.numAttributes(); i++) {
			TreeMap<Double, Integer> tmpTreeMap = null;
			tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestParentsList.add(i, tmpTreeMap);
			tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestChildrenList.add(i, tmpTreeMap);
			tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestParentsAndChildrenList.add(i, tmpTreeMap);
		}
		
		//map<entropy, rule(string)>
		TreeMap<Double, String> entropyRuleMap = new TreeMap<Double, String>();

		//map<entropy, rule(attributeChildIndex <- attributeParentIndex)>
		TreeMap<Double, Entry<Integer,Integer>> entropyChildFromParentMap = new TreeMap<Double, Entry<Integer,Integer>>();

		//map<entropy, rule(attributeParentIndex -> attributeChildIndex)>
		TreeMap<Double, Entry<Integer,Integer>> entropyParentToChildMap = new TreeMap<Double, Entry<Integer,Integer>>();
		
		//Idea 1
		//map<entropy, addParent(whichAttribute, toAdd)>
		TreeMap<Double, Entry<Integer,Integer>> entropyBestRuleMap = new TreeMap<Double, Entry<Integer,Integer>>();
		
		// calculate conditional entropy for contingency tables
		for (int i = 0; i < instances.numAttributes(); i++) {
			for (int j = 0; j < i; j++) {
				double entropyConditionedOnRows = ContingencyTables.entropyConditionedOnRows(attributeMatrix[i][j]);
				double entropyConditionedOnColumns = ContingencyTables.entropyConditionedOnColumns(attributeMatrix[i][j]);

				double lowestEntropy = (entropyConditionedOnRows < entropyConditionedOnColumns) ? 
						entropyConditionedOnRows : 
						entropyConditionedOnColumns;
				
				//save current rule
				String arc = (entropyConditionedOnRows < entropyConditionedOnColumns) ? 
						instances.attribute(j).name() + " <- " + instances.attribute(i).name() : 
						instances.attribute(i).name() + " <- " + instances.attribute(j).name();
				entropyRuleMap.put(lowestEntropy, arc);
				
				//Idea 1
				entropyBestRuleMap.put(entropyConditionedOnRows, new AbstractMap.SimpleEntry<Integer,Integer>(j,i));
				entropyBestRuleMap.put(entropyConditionedOnColumns, new AbstractMap.SimpleEntry<Integer,Integer>(i,j));
						
				if (entropyConditionedOnRows < entropyConditionedOnColumns) {
					attributeBestParentsList.get(j).put(lowestEntropy, i);
					attributeBestChildrenList.get(i).put(lowestEntropy, j);
					entropyChildFromParentMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer,Integer>(j,i));
					entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer,Integer>(i,j));
				} else {
					attributeBestParentsList.get(i).put(lowestEntropy, j);
					attributeBestChildrenList.get(j).put(lowestEntropy, i);
					entropyChildFromParentMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer,Integer>(i,j));
					entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer,Integer>(j,i));
				}
			}
		}

		//Greedy algorithm: add parents from the full list of rules (sorted)
		//if true not usable, if false (default) usable
		boolean blackList[] = new boolean[instances.numAttributes()];
		
		for (Double key : entropyBestRuleMap.keySet()) {
			Map.Entry<Integer,Integer> entry = entropyBestRuleMap.get(key);
			
			//add parents
			//bayesNet.getParentSet(entry.getKey()).addParent(entry.getValue(), instances);
			if(!blackList[entry.getKey()] && !blackList[entry.getValue()]){
				bayesNet.getParentSet(entry.getKey()).addParent(entry.getValue(), instances);
				blackList[entry.getKey()] = true;
				blackList[entry.getValue()] = true;
			}
			
//			if(!blackList[entry.getKey()]){
//				bayesNet.getParentSet(entry.getKey()).addParent(entry.getValue(), instances);
//				blackList[entry.getKey()] = true;
//			}
		}
		
//		System.out.println("\nBestParentsAndChildrenFullListSearch.search(): complete");
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
}