package il.ac.openu.bestparents;

import weka.core.ContingencyTables;

/**
 * Tests conditional entropy for attributes
 * 
 * @author Andrew Kreimer
 *
 */
public class ContingencyTablesTest {
	public static void main(String[] args) {
		double matrix[][] = {
				{2.0/16, 1.0/16, 1.0/16, 0.25},
				{0, 3.0/16, 1.0/16, 0},
				{1.0/32, 1.0/32, 1.0/16, 0},
				{1.0/32, 1.0/32, 1.0/16, 0}
		};
		
		System.out.println("ContingencyTables.entropyConditionedOnRows(matrix): " + ContingencyTables.entropyConditionedOnRows(matrix));
		System.out.println("ContingencyTables.entropyConditionedOnColumns(matrix): " + ContingencyTables.entropyConditionedOnColumns(matrix));
	}
}
