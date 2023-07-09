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
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.TreeMap;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.ContingencyTables;
import weka.core.Instances;

/**
 * Bes parents and children recursive search.
 *
 * @author Andrew Kreimer
 */
public class BestParentsAndChildrenRecursiveSearch extends SearchAlgorithm {

  private static final long serialVersionUID = 2467629575499347683L;

  private int maxNrOfChildren;
  private double[][][][] attributeMatrix;

  private List<TreeMap<Double, Integer>> attributeBestParentsList = Collections.emptyList();
  private List<TreeMap<Double, Integer>> attributeBestChildrenList = Collections.emptyList();

  private BayesNet bayesNet;
  private Instances instances;
  private boolean[] blackList;

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

    // for each attribute with index i: map<entropy, parent index>, keeping
    // the map sorted
    attributeBestParentsList = new ArrayList<>();
    attributeBestChildrenList = new ArrayList<>();

    // allocate
    for (var i = 0; i < instances.numAttributes(); i++) {
      TreeMap<Double, Integer> tmpTreeMap = null;
      tmpTreeMap = new TreeMap<>();
      attributeBestParentsList.add(i, tmpTreeMap);
      tmpTreeMap = new TreeMap<>();
      attributeBestChildrenList.add(i, tmpTreeMap);
    }

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

        // save current rule
        if (entropyConditionedOnRows < entropyConditionedOnColumns) {
          attributeBestParentsList.get(j).put(lowestEntropy, i);
          attributeBestChildrenList.get(i).put(lowestEntropy, j);
        } else {
          attributeBestParentsList.get(i).put(lowestEntropy, j);
          attributeBestChildrenList.get(j).put(lowestEntropy, i);
        }
      }
    }

    addBestParentsAndChildrenIterative(bayesNet, instances, instances.numAttributes() - 1);
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

  /**
   * Adds nodes.
   *
   * <p>Recursive algorithm: start with class by adding best parent and child, expand for child and
   * parent the same algorithm till no more attributes left.
   *
   * @param bayesNet
   * @param instances
   * @param i attribute
   */
  private void addBestParentsAndChildren(
      BayesNet bayesNet, Instances instances, int i) {
    this.bayesNet = bayesNet;
    this.instances = instances;

    TreeMap<Double, Integer> tmpBestChildrenMap = attributeBestChildrenList.get(i);
    TreeMap<Double, Integer> tmpBestParentsMap = attributeBestParentsList.get(i);

    double bestChildKey = Double.POSITIVE_INFINITY; // +infinity
    double bestParentKey = Double.POSITIVE_INFINITY; // +infinity

    if (tmpBestChildrenMap.keySet().toArray().length != 0) {
      bestChildKey = (double) tmpBestChildrenMap.keySet().toArray()[0];
    }

    if (tmpBestParentsMap.keySet().toArray().length != 0) {
      bestParentKey = (double) tmpBestParentsMap.keySet().toArray()[0];
    }

    Integer bestChild = tmpBestChildrenMap.get(bestChildKey);
    Integer bestParent = tmpBestParentsMap.get(bestParentKey);

    var expandToChild = false;
    var expandToParent = false;

    // order matters, best child or parent could be stolen because of
    // execution order!
    var path = new PathData();
    path.setI(i);
    path.setTmpBestChildrenMap(tmpBestChildrenMap);
    path.setTmpBestParentsMap(tmpBestParentsMap);
    path.setBestChildKey(bestChildKey);
    path.setBestParentKey(bestParentKey);
    path.setBestChild(bestChild);
    path.setBestParent(bestParent);
    path.setExpandToChild(expandToChild);
    path.setExpandToParent(expandToParent);

    if (bestChildKey < bestParentKey) {
      expandChildPath(path);
    } else {
      expandParentPath(path);
    }
  }

  /**
   * Expands parent's path.
   *
   * @param path
   */
  private void expandParentPath(PathData path) {
    boolean expandToParent = path.isExpandToParent();
    boolean expandToChild = path.isExpandToChild();
    if (path.getBestParent() != null && !blackList[path.getBestParent()]) {
      bayesNet.getParentSet(path.getI()).addParent(path.getBestParent(), instances);
      blackList[path.getBestParent()] = true;
      expandToParent = true;
      path.getTmpBestParentsMap().remove(path.getBestParentKey());
    }

    if (path.getBestChild() != null && !blackList[path.getBestChild()]) {
      bayesNet.getParentSet(path.getBestChild()).addParent(path.getI(), instances);
      blackList[path.getBestChild()] = true;
      expandToChild = true;
      path.getTmpBestChildrenMap().remove(path.getBestChildKey());
    }

    if (expandToParent) {
      addBestParentsAndChildren(bayesNet, instances, path.getBestParent());
    }

    if (expandToChild) {
      addBestParentsAndChildren(bayesNet, instances, path.getBestChild());
    }
  }

  /**
   * Expands child's path.
   *
   * @param path
   */
  private void expandChildPath(PathData path) {
    boolean expandToParent = path.isExpandToParent();
    boolean expandToChild = path.isExpandToChild();
    if (path.getBestChild() != null && !blackList[path.getBestChild()]) {
      bayesNet.getParentSet(path.getBestChild()).addParent(path.getI(), instances);
      blackList[path.getBestChild()] = true;
      expandToChild = true;
      path.getTmpBestChildrenMap().remove(path.getBestChildKey());
    }

    if (path.getBestParent() != null && !blackList[path.getBestParent()]) {
      bayesNet.getParentSet(path.getI()).addParent(path.getBestParent(), instances);
      blackList[path.getBestParent()] = true;
      expandToParent = true;
      path.getTmpBestParentsMap().remove(path.getBestParentKey());
    }

    if (expandToChild) {
      addBestParentsAndChildren(bayesNet, instances, path.getBestChild());
    }

    if (expandToParent) {
      addBestParentsAndChildren(bayesNet, instances, path.getBestParent());
    }
  }

  /**
   * Adds nodes.
   *
   * <p>Iterative algorithm: start with class by adding best parent and child, expand for child and
   * parent the same algorithm till no more attributes left. Use queue instead of recursive
   * expansion.
   *
   * @param bayesNet
   * @param instances
   * @param startingAttribute
   */
  public void addBestParentsAndChildrenIterative(
      BayesNet bayesNet, Instances instances, int startingAttribute) {
    // keeping the expansion order is important, expand to the lowest
    // conditional entropies first
    TreeMap<Double, Integer> expansionOrder = new TreeMap<>();

    // blackList: mark Class as used
    blackList = new boolean[instances.numAttributes()];
    blackList[startingAttribute] = true;

    // expansion queue
    Queue<Integer> queue = new LinkedList<>();

    // start with class
    queue.add(startingAttribute);

    // while there are attributes to expand
    while (!queue.isEmpty()) {
      // current attribute
      Integer attribute = queue.remove();
      blackList[attribute] = true;

      // add best parents
      addBestParents(bayesNet, instances, expansionOrder, blackList, attribute);

      // add best children
      addBestChildren(bayesNet, instances, expansionOrder, blackList, attribute);

      // if queue is empty, proceed with expansion by order
      if (queue.isEmpty()) {
        queue = new LinkedList<>();
        queue.addAll(expansionOrder.values());
        expansionOrder = new TreeMap<>();
      }
    }
  }

  /**
   * Adds best children.
   *
   * @param bayesNet
   * @param instances
   * @param expansionOrder
   * @param blackList
   * @param attribute
   */
  private void addBestChildren(
      BayesNet bayesNet,
      Instances instances,
      TreeMap<Double, Integer> expansionOrder,
      boolean[] blackList,
      Integer attribute) {
    for (var i = 0; i < getMaxNrOfChildren(); i++) {
      Object[] arr = attributeBestChildrenList.get(attribute).keySet().toArray();

      // if there are rules
      if (arr.length == 0) {
        break;
      }

      Double key = (Double) arr[i];
      Integer val = attributeBestChildrenList.get(attribute).get(key);

      if (i < arr.length && !blackList[val]
      /* && bayesNet.getParentSet(val).getNrOfParents() < getMaxNrOfParents() */ ) {
        bayesNet.getParentSet(val).addParent(i, instances);
        expansionOrder.put(key, val);
        blackList[val] = true;
      }
    }
  }

  /**
   * Adds best parents.
   *
   * @param bayesNet
   * @param instances
   * @param expansionOrder
   * @param blackList
   * @param attribute
   */
  private void addBestParents(
      BayesNet bayesNet,
      Instances instances,
      TreeMap<Double, Integer> expansionOrder,
      boolean[] blackList,
      Integer attribute) {
    for (var i = 0; i < getMaxNrOfParents(); i++) {
      Object[] arr = attributeBestParentsList.get(attribute).keySet().toArray();

      // if there are rules
      if (arr.length == 0) {
        break;
      }

      Double key = (Double) arr[i];
      Integer val = attributeBestParentsList.get(attribute).get(key);

      if (i < arr.length && !blackList[val]
      /* && bayesNet.getParentSet(attribute).getNrOfParents() < getMaxNrOfParents() */ ) {
        bayesNet.getParentSet(attribute).addParent(val, instances);
        expansionOrder.put(key, val);
        blackList[val] = true;
      }
    }
  }
}
