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
import java.util.Map.Entry;
import java.util.TreeMap;

import il.ac.openu.bestparents.util.NewBNUtils;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.ContingencyTables;
import weka.core.Instances;

/**
 * @author Andrew Kreimer - algonell.com
 */
public class BestParentsAndChildrenSearch extends SearchAlgorithm{

	private static final long serialVersionUID = 8139091196984853152L;
	
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

		//Greedy algorithm: for each attribute take best child or parent, having the lower entropy
		//if true not usable, if false (default) usable
		boolean parentsBlackList[] = new boolean[instances.numAttributes()];
		boolean childrenBlackList[] = new boolean[instances.numAttributes()];
		
		for (int i = 0; i < instances.numAttributes(); i++) {
			TreeMap<Double, Integer> tmpBestChildrenMap = attributeBestChildrenList.get(i);
			TreeMap<Double, Integer> tmpBestParentsMap = attributeBestParentsList.get(i);
			int numOfAddedRules = 0;
			
			double bestChildKey = Double.MAX_VALUE;//+infinity
			double bestParentKey = Double.MAX_VALUE;//+infinity
			
			if(tmpBestChildrenMap.keySet().toArray().length != 0){
				bestChildKey = (double)tmpBestChildrenMap.keySet().toArray()[0];
			}
			
			if(tmpBestParentsMap.keySet().toArray().length != 0){
				bestParentKey = (double)tmpBestParentsMap.keySet().toArray()[0];
			}
			
//			System.out.println("BestParentsAndChildrenOriginalSearch " + i + ": " + bestChildKey + "(" + tmpBestChildrenMap.get(bestChildKey) +  ") < " + bestParentKey + "(" + tmpBestParentsMap.get(bestParentKey) + ")");
			
			if(bestChildKey < bestParentKey){//if child is better than parent (entropies comparison)
				int numOfParentsForCurrentChild = bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).getNrOfParents();
				
				if (
						numOfAddedRules < getMaxNrOfChildren() &&
						numOfParentsForCurrentChild < getMaxNrOfParents() &&
						numOfAddedRules < tmpBestChildrenMap.size() &&
						NewBNUtils.countNumOfChildren(bayesNet, instances, i) < getMaxNrOfChildren() &&
						!bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).contains(i) &&
						!childrenBlackList[tmpBestChildrenMap.get(bestChildKey)] &&
						tmpBestParentsMap.get(bestParentKey) != null &&
						!parentsBlackList[tmpBestParentsMap.get(bestParentKey)]
					){
					bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).addParent(i, instances);
					numOfAddedRules++;
					childrenBlackList[tmpBestChildrenMap.get(bestChildKey)] = true;
//					System.out.println("BestParentsAndChildrenOriginalSearch " + i + " -> " + tmpBestChildrenMap.get(bestChildKey));
				}
				
				//bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).addParent(i, instances);
			}else{
				if (
						numOfAddedRules < getMaxNrOfParents() && 
						numOfAddedRules < tmpBestParentsMap.size() &&
						NewBNUtils.countNumOfChildren(bayesNet, instances, tmpBestParentsMap.get(bestParentKey)) < getMaxNrOfChildren() &&
						!bayesNet.getParentSet(i).contains(tmpBestParentsMap.get(bestParentKey)) &&
						!parentsBlackList[tmpBestParentsMap.get(bestParentKey)] &&
						tmpBestChildrenMap.get(bestChildKey) != null &&
						!childrenBlackList[tmpBestChildrenMap.get(bestChildKey)]
					){
					bayesNet.getParentSet(i).addParent(tmpBestParentsMap.get(bestParentKey), instances);
					numOfAddedRules++;
					parentsBlackList[tmpBestParentsMap.get(bestParentKey)] = true;
//					System.out.println("BestParentsAndChildrenOriginalSearch " + tmpBestParentsMap.get(bestParentKey) + " <- " + i);
				}
				
				//bayesNet.getParentSet(i).addParent(tmpBestParentsMap.get(bestParentKey), instances);
			}
		}
		
//		System.out.println("\nBestParentsAndChildrenOriginalSearch: complete");
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