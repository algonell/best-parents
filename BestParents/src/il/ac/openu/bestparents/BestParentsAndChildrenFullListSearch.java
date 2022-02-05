/*
 * This program is free software: you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program. If
 * not, see <http://www.gnu.org/licenses/>.
 */
package il.ac.openu.bestparents;

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
 * Best parents and children search.
 *
 * @author Andrew Kreimer
 */
public class BestParentsAndChildrenFullListSearch extends SearchAlgorithm {

  private static final long serialVersionUID = -6875216741076169820L;

  private int maxNrOfChildren;
  private double[][][][] attributeMatrix;

  /**
   * Performs path search.
   *
   * @param bayesNet the network
   * @param instances the data to work with
   */
  @Override
  public void search(BayesNet bayesNet, Instances instances) {
    // contingency table for each attribute X attribute matrix
    attributeMatrix = new double[instances.numAttributes()][instances.numAttributes()][][];

    // allocate
    allocate(instances);

    // count instantiations
    count(instances);

    // for each attribute with index i: map<entropy, parent index>, keeping the map sorted
    ArrayList<TreeMap<Double, Integer>> attributeBestParentsList = new ArrayList<>();
    ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList = new ArrayList<>();
    ArrayList<TreeMap<Double, Integer>> attributeBestParentsAndChildrenList = new ArrayList<>();

    // allocate
    for (var i = 0; i < instances.numAttributes(); i++) {
      TreeMap<Double, Integer> tmpTreeMap = null;
      tmpTreeMap = new TreeMap<>();
      attributeBestParentsList.add(i, tmpTreeMap);
      tmpTreeMap = new TreeMap<>();
      attributeBestChildrenList.add(i, tmpTreeMap);
      tmpTreeMap = new TreeMap<>();
      attributeBestParentsAndChildrenList.add(i, tmpTreeMap);
    }

    // map<entropy, rule(string)>
    TreeMap<Double, String> entropyRuleMap = new TreeMap<>();

    // map<entropy, rule(attributeChildIndex <- attributeParentIndex)>
    TreeMap<Double, Entry<Integer, Integer>> entropyChildFromParentMap = new TreeMap<>();

    // map<entropy, rule(attributeParentIndex -> attributeChildIndex)>
    TreeMap<Double, Entry<Integer, Integer>> entropyParentToChildMap = new TreeMap<>();

    // Idea 1
    // map<entropy, addParent(whichAttribute, toAdd)>
    TreeMap<Double, Entry<Integer, Integer>> entropyBestRuleMap = new TreeMap<>();

    // calculate conditional entropy for contingency tables
    calculateContingencyTables(
        instances,
        attributeBestParentsList,
        attributeBestChildrenList,
        entropyRuleMap,
        entropyChildFromParentMap,
        entropyParentToChildMap,
        entropyBestRuleMap);

    // Greedy algorithm: add parents from the full list of rules (sorted)
    // if true not usable, if false (default) usable
    var blackList = new boolean[instances.numAttributes()];

    for (Entry<Double, Entry<Integer, Integer>> entry : entropyBestRuleMap.entrySet()) {
      Map.Entry<Integer, Integer> value = entry.getValue();

      // add parents
      if (!blackList[value.getKey()] && !blackList[value.getValue()]) {
        bayesNet.getParentSet(value.getKey()).addParent(value.getValue(), instances);
        blackList[value.getKey()] = true;
        blackList[value.getValue()] = true;
      }
    }
  }

  /**
   * Calculate conditional entropies.
   *
   * @param instances
   * @param attributeBestParentsList
   * @param attributeBestChildrenList
   * @param entropyRuleMap
   * @param entropyChildFromParentMap
   * @param entropyParentToChildMap
   * @param entropyBestRuleMap
   */
  private void calculateContingencyTables(
      Instances instances,
      ArrayList<TreeMap<Double, Integer>> attributeBestParentsList,
      ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList,
      TreeMap<Double, String> entropyRuleMap,
      TreeMap<Double, Entry<Integer, Integer>> entropyChildFromParentMap,
      TreeMap<Double, Entry<Integer, Integer>> entropyParentToChildMap,
      TreeMap<Double, Entry<Integer, Integer>> entropyBestRuleMap) {
    for (var i = 0; i < instances.numAttributes(); i++) {
      for (var j = 0; j < i; j++) {
        double entropyConditionedOnRows =
            ContingencyTables.entropyConditionedOnRows(attributeMatrix[i][j]);
        double entropyConditionedOnColumns =
            ContingencyTables.entropyConditionedOnColumns(attributeMatrix[i][j]);

        double lowestEntropy =
            (entropyConditionedOnRows < entropyConditionedOnColumns)
                ? entropyConditionedOnRows
                : entropyConditionedOnColumns;

        // save current rule
        String arc =
            (entropyConditionedOnRows < entropyConditionedOnColumns)
                ? instances.attribute(j).name() + " <- " + instances.attribute(i).name()
                : instances.attribute(i).name() + " <- " + instances.attribute(j).name();
        entropyRuleMap.put(lowestEntropy, arc);

        // Idea 1
        entropyBestRuleMap.put(entropyConditionedOnRows, new AbstractMap.SimpleEntry<>(j, i));
        entropyBestRuleMap.put(entropyConditionedOnColumns, new AbstractMap.SimpleEntry<>(i, j));

        if (entropyConditionedOnRows < entropyConditionedOnColumns) {
          attributeBestParentsList.get(j).put(lowestEntropy, i);
          attributeBestChildrenList.get(i).put(lowestEntropy, j);
          entropyChildFromParentMap.put(lowestEntropy, new AbstractMap.SimpleEntry<>(j, i));
          entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<>(i, j));
        } else {
          attributeBestParentsList.get(i).put(lowestEntropy, j);
          attributeBestChildrenList.get(j).put(lowestEntropy, i);
          entropyChildFromParentMap.put(lowestEntropy, new AbstractMap.SimpleEntry<>(i, j));
          entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<>(j, i));
        }
      }
    }
  }

  /**
   * Counts occurrences.
   *
   * @param instances
   */
  private void count(Instances instances) {
    for (var n = 0; n < instances.numInstances(); n++) {
      for (var i = 0; i < instances.numAttributes(); i++) {
        for (var j = 0; j < i; j++) {
          int iAttrIndex = (int) instances.instance(n).value(i);
          int jAttrIndex = (int) instances.instance(n).value(j);
          attributeMatrix[i][j][iAttrIndex][jAttrIndex]++;
        }
      }
    }
  }

  /**
   * Allocates memory.
   *
   * @param instances
   */
  private void allocate(Instances instances) {
    for (var j = 0; j < instances.numAttributes(); j++) {
      for (var k = 0; k < j; k++) {
        attributeMatrix[j][k] =
            new double[instances.attribute(j).numValues()][instances.attribute(k).numValues()];
      }
    }
  }

  /** Sets the max number of parents. */
  public void setMaxNrOfParents(int nMaxNrOfParents) {
    m_nMaxNrOfParents = nMaxNrOfParents;
  }

  /** Gets the max number of parents. */
  public int getMaxNrOfParents() {
    return m_nMaxNrOfParents;
  }

  /** Sets the max number of children. */
  public void setMaxNrOfChildren(int nMaxNrOfChildren) {
    maxNrOfChildren = nMaxNrOfChildren;
  }

  /** Gets the max number of children. */
  public int getMaxNrOfChildren() {
    return maxNrOfChildren;
  }
}
