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
public class BestParentsAndChildrenSearch extends SearchAlgorithm {

  private static final long serialVersionUID = 8139091196984853152L;

  private int maxNrOfChildren;
  private double[][][][] attributeMatrix;

  ArrayList<TreeMap<Double, Integer>> attributeBestParentsList;
  ArrayList<TreeMap<Double, Integer>> attributeBestChildrenList;
  ArrayList<TreeMap<Double, Integer>> attributeBestParentsAndChildrenList;

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
    attributeBestParentsList = new ArrayList<>();
    attributeBestChildrenList = new ArrayList<>();
    attributeBestParentsAndChildrenList = new ArrayList<>();

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

    // calculate conditional entropy for contingency tables
    calculateContingencyTables(
        instances, entropyRuleMap, entropyChildFromParentMap, entropyParentToChildMap);

    // Greedy algorithm: for each attribute take best child or parent, having the lower entropy
    // if true not usable, if false (default) usable
    var parentsBlackList = new boolean[instances.numAttributes()];
    var childrenBlackList = new boolean[instances.numAttributes()];

    for (var i = 0; i < instances.numAttributes(); i++) {
      TreeMap<Double, Integer> tmpBestChildrenMap = attributeBestChildrenList.get(i);
      TreeMap<Double, Integer> tmpBestParentsMap = attributeBestParentsList.get(i);
      var numOfAddedRules = 0;

      double bestChildKey = Double.MAX_VALUE; // +infinity
      double bestParentKey = Double.MAX_VALUE; // +infinity

      if (tmpBestChildrenMap.keySet().toArray().length != 0) {
        bestChildKey = (double) tmpBestChildrenMap.keySet().toArray()[0];
      }

      if (tmpBestParentsMap.keySet().toArray().length != 0) {
        bestParentKey = (double) tmpBestParentsMap.keySet().toArray()[0];
      }

      // if child is better than parent (entropies comparison)
      if (bestChildKey < bestParentKey) {
        int numOfParentsForCurrentChild =
            bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).getNrOfParents();

        if (numOfAddedRules < getMaxNrOfChildren()
            && numOfParentsForCurrentChild < getMaxNrOfParents()
            && numOfAddedRules < tmpBestChildrenMap.size()
            && BnUtils.countNumOfChildren(bayesNet, instances, i) < getMaxNrOfChildren()
            && !bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).contains(i)
            && !childrenBlackList[tmpBestChildrenMap.get(bestChildKey)]
            && tmpBestParentsMap.get(bestParentKey) != null
            && !parentsBlackList[tmpBestParentsMap.get(bestParentKey)]) {
          bayesNet.getParentSet(tmpBestChildrenMap.get(bestChildKey)).addParent(i, instances);
          childrenBlackList[tmpBestChildrenMap.get(bestChildKey)] = true;
        }
      } else if (numOfAddedRules < getMaxNrOfParents()
          && numOfAddedRules < tmpBestParentsMap.size()
          && BnUtils.countNumOfChildren(bayesNet, instances, tmpBestParentsMap.get(bestParentKey))
              < getMaxNrOfChildren()
          && !bayesNet.getParentSet(i).contains(tmpBestParentsMap.get(bestParentKey))
          && !parentsBlackList[tmpBestParentsMap.get(bestParentKey)]
          && tmpBestChildrenMap.get(bestChildKey) != null
          && !childrenBlackList[tmpBestChildrenMap.get(bestChildKey)]) {
        bayesNet.getParentSet(i).addParent(tmpBestParentsMap.get(bestParentKey), instances);
        parentsBlackList[tmpBestParentsMap.get(bestParentKey)] = true;
      }
    }
  }

  /**
   * Calculates conditional entropies.
   *
   * @param instances
   * @param entropyRuleMap
   * @param entropyChildFromParentMap
   * @param entropyParentToChildMap
   */
  private void calculateContingencyTables(
      Instances instances,
      TreeMap<Double, String> entropyRuleMap,
      TreeMap<Double, Entry<Integer, Integer>> entropyChildFromParentMap,
      TreeMap<Double, Entry<Integer, Integer>> entropyParentToChildMap) {
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
