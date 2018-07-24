package com.oracle.graphpipe;

import junit.framework.TestCase;

public class NativeTensorTest extends TestCase {
    public void testRectAryPrimitive() {
        assertFalse(NativeTensor.isRectAry(4));
    }
    
    public void testRectAryRank2() {
        // Array of rank 2, dims [4][2].
        int ary[][] = {{1, 2}, {3, 4}, {6, 7}};
        assertTrue(NativeTensor.isRectAry(ary));
    }
    
    public void testRectAryRank3() {
        // Array of rank 3, dims [2][3][2].
        int ary[][][] = {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}};
        assertTrue(NativeTensor.isRectAry(ary));
    } 
   
    public void testRectAryDiffDims() {
        // Sub-dims differ ([1] and [2]).
        int ary[][] = {{1}, {2, 3}};
        assertFalse(NativeTensor.isRectAry(ary));
    }
    
    public void testRectAryDiffDimsSameSizes() {
        // Sub-dims [3][2] and [2][3]. Both have total size 6.
        int ary[][][] = {{{1, 2}, {3, 4}, {6, 7}}, {{1, 2, 3}, {4, 5, 6}}};
        assertFalse(NativeTensor.isRectAry(ary));
    }

    public void testRectAryDiffRanks() {
        // Sub-dims [1][0] and [0].
        int ary[][][] = {{{}}, {}};
        assertFalse(NativeTensor.isRectAry(ary));
    }
}
