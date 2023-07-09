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

import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.TreeMap;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.ContingencyTables;
import weka.core.Instances;

/**
 * Best parents search.
 *
 * @author Andrew Kreimer
 */
public class BestParentsSearch extends SearchAlgorithm {

  private static final long serialVersionUID = -8315181456697597693L;

  /**
   * Performs path search.
   *
   * @param bayesNet the network
   * @param instances the data to work with
   */
  @Override
  public void search(BayesNet bayesNet, Instances instances) {
    // contingency table for each [attribute X attribute] matrix
    double[][][][] attributeMatrix = allocateAttributeMatrix(instances);

    // count instantiations
    countInstantiations(instances, attributeMatrix);

    // for each attribute with index i: map<entropy, parent index>, keeping the map sorted
    ArrayList<TreeMap<Double, Integer>> attributeBestParentsList = allocateAttributeMaps(instances);

    findBestParents(instances, attributeMatrix, attributeBestParentsList);

    // add good parents, for each attribute, bounded by maxNumberOfParents
    addBestRules(bayesNet, instances, attributeBestParentsList);
  }

  private void addBestRules(
      BayesNet bayesNet,
      Instances instances,
      ArrayList<TreeMap<Double, Integer>> attributeBestParentsList) {
    for (var i = 0; i < instances.numAttributes(); i++) {
      TreeMap<Double, Integer> tmpTreeMap = attributeBestParentsList.get(i);
      var numOfAddedRules = 0;

      for (Entry<Double, Integer> entry : tmpTreeMap.entrySet()) {
        int value = entry.getValue();

        if (numOfAddedRules < getMaxNrOfParents()
            && numOfAddedRules < tmpTreeMap.size()
            &&
            // avoid parents with several children
            BnUtils.countNumOfChildren(bayesNet, instances, value) < getMaxNrOfParents()
            && !bayesNet.getParentSet(i).contains(value)) {
          bayesNet.getParentSet(i).addParent(value, instances);
          numOfAddedRules++;
        }
      }
    }
  }

  /**
   * Finds the best parents for each attribute by conditional entropy and greedy algorithm.
   *
   * @param instances
   * @param attributeMatrix
   * @param attributeBestParentsList
   */
  private void findBestParents(
      Instances instances,
      double[][][][] attributeMatrix,
      ArrayList<TreeMap<Double, Integer>> attributeBestParentsList) {
    // calculate conditional entropy for contingency tables
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

        // best rule
        if (entropyConditionedOnRows < entropyConditionedOnColumns) {
          attributeBestParentsList.get(j).put(lowestEntropy, i);
        } else {
          attributeBestParentsList.get(i).put(lowestEntropy, j);
        }
      }
    }
  }

  /**
   * Allocates List of Map: Map of best parents for each attribute.
   *
   * @param instances
   */
  private ArrayList<TreeMap<Double, Integer>> allocateAttributeMaps(Instances instances) {
    ArrayList<TreeMap<Double, Integer>> attributeBestParentsList = new ArrayList<>();

    // allocate
    for (var i = 0; i < instances.numAttributes(); i++) {
      TreeMap<Double, Integer> tmpTreeMap = new TreeMap<>();
      attributeBestParentsList.add(i, tmpTreeMap);
    }

    return attributeBestParentsList;
  }

  /**
   * Counts instances for each attribute and category.
   *
   * @param instances
   * @param attributeMatrix
   */
  private void countInstantiations(Instances instances, double[][][][] attributeMatrix) {
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
   * Allocates contingency matrix for each attribute-attribute pair.
   *
   * @param instances
   */
  private double[][][][] allocateAttributeMatrix(Instances instances) {
    var attributeMatrix = new double[instances.numAttributes()][instances.numAttributes()][][];

    // allocate
    for (var j = 0; j < instances.numAttributes(); j++) {
      for (var k = 0; k < j; k++) {
        attributeMatrix[j][k] =
            new double[instances.attribute(j).numValues()][instances.attribute(k).numValues()];
      }
    }

    return attributeMatrix;
  }

  /** Sets the max number of parents. */
  public void setMaxNrOfParents(int nMaxNrOfParents) {
    m_nMaxNrOfParents = nMaxNrOfParents;
  }

  /** Gets the max number of parents. */
  public int getMaxNrOfParents() {
    return m_nMaxNrOfParents;
  }
}
