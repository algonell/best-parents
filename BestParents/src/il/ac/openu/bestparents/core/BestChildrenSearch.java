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
public class BestChildrenSearch extends SearchAlgorithm{

	private static final long serialVersionUID = 1032285588625105530L;
	
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

		// for each attribute with index i: map<entropy, child index>, keeping the map sorted
		ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList = new ArrayList<TreeMap<Double, Integer>>();
		
		//allocate
		for (int i = 0; i < instances.numAttributes(); i++) {
			TreeMap<Double, Integer> tmpTreeMap = new TreeMap<Double, Integer>();
			attributeBestChildrenList.add(i, tmpTreeMap);
		}
		
		//map<entropy, rule(string)>
		TreeMap<Double, String> entropyRuleMap = new TreeMap<Double, String>();

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
						instances.attribute(i).name() + " -> " + instances.attribute(j).name() : 
						instances.attribute(j).name() + " -> " + instances.attribute(i).name();
				entropyRuleMap.put(lowestEntropy, arc);
				
				if (entropyConditionedOnRows < entropyConditionedOnColumns) {
					attributeBestChildrenList.get(i).put(lowestEntropy, j);
					entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer,Integer>(i,j));
				} else {
					attributeBestChildrenList.get(j).put(lowestEntropy, i);
					entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<Integer,Integer>(j,i));
				}
			}
		}

		for (int i = 0; i < instances.numAttributes(); i++) {
			TreeMap<Double, Integer> tmpTreeMap = attributeBestChildrenList.get(i);
			int numOfAddedRules = 0;
			
			for (Double key : tmpTreeMap.keySet()) {
				int numOfParentsForCurrentChild = bayesNet.getParentSet(tmpTreeMap.get(key)).getNrOfParents();
				if (numOfAddedRules < getMaxNrOfChildren() &&
						numOfParentsForCurrentChild < getMaxNrOfChildren() &&
						numOfAddedRules < tmpTreeMap.size() &&
						NewBNUtils.countNumOfChildren(bayesNet, instances, i) < getMaxNrOfChildren() &&
						!bayesNet.getParentSet(tmpTreeMap.get(key)).contains(i)){
					bayesNet.getParentSet(tmpTreeMap.get(key)).addParent(i, instances);
					numOfAddedRules++;
				}
			}
		}

		// print sorted rules for each attribute (best children)
//		System.out.println("\n--------------------------------------------------");
//		System.out.println("print sorted rules for each attribute (best children):");
//		int i = 0 ;
//		for (TreeMap<Double, Integer> tmpTreeMap : attributeBestChildrenList) {
//			System.out.println("i: " + i);
//			for (Double key : tmpTreeMap.keySet()) {
//				System.out.println("Entropy: " + key + ", " + tmpTreeMap.get(key));
//			}
//			i++;
//			System.out.println("");
//		}
		
//		System.out.println("\nBNBestChildrenSearch.search(): complete");
	} // buildStructure
	
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
