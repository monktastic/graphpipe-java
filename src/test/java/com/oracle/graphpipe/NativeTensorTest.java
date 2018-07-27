package com.oracle.graphpipe;

import junit.framework.TestCase;
import org.junit.Assert;

import java.util.Arrays;

public class NativeTensorTest extends TestCase {
    public void testCtorDiffDims() {
        try {
            // Sub-dims differ ([1] and [2]).
            int ary[][] = {{1}, {2, 3}};
            new NativeTensor(ary);
            fail("Should have failed");
        } catch (IllegalArgumentException e) {
            assertTrue(e.toString().contains("not rectangular"));
        }
    }
    
    public void testCtorDiffDimsSameSizes() {
        try {
            // Sub-dims [3][2] and [2][3]. Both have total size 6.
            int ary[][][] = {{{1, 2}, {3, 4}, {6, 7}}, {{1, 2, 3}, {4, 5, 6}}};
            new NativeTensor(ary);
            fail("Should have failed");
        } catch (IllegalArgumentException e) {
            assertTrue(e.toString().contains("not rectangular"));
        }
    }

    public void testCtorDiffRanks() {
        try {
            // Sub-dims [1][1] and [0].
            int ary[][][] = {{{1}}, {}};
            new NativeTensor(ary);
            fail("Should have failed");
        } catch (IllegalArgumentException e) {
            assertTrue(e.toString().contains("not rectangular"));
        }
    }

    public void testCtorEmptyAry() {
        try {
            int ary[][] = {{}, {}};
            new NativeTensor(ary);
            fail("Should have failed");
        } catch (ArrayIndexOutOfBoundsException e) {
        }
    }
    
    public void testCtorRank1() {
        int ary[] = {1, 2, 3};
        new NativeTensor(ary);
    }
    
    public void testCtorRank2() {
        byte ary[][] = {{1, 2}, {3, 4}, {5, 6}};
        NativeTensor nt = new NativeTensor(ary);
        assertEquals(6, nt.data.array().length);
    }
    
    public void testCtorRank3() {
        short ary[][][] = {
                {{0x0001, 0x0020}, {0x0300, 0x4000}}, 
                {{0x000A, 0x00B0}, {0x0C00, (short)0xD000}}};
        NativeTensor nt = new NativeTensor(ary);
        byte[] expected = {
                0x01, 0x00, 0x20, 0x00, 0x00, 0x03, 0x00, 0x40,
                0x0A, 0x00, (byte)0xB0, 0x00, 0x00, 0x0C, 0x00, (byte)0xD0,
        };
        Assert.assertArrayEquals(expected, nt.data.array());
    }
    
    public void testCtorStringData() {
        String ary[][] = {{"a", "bc"}, {"def", "ghij"}};
        NativeTensor nt = new NativeTensor(ary);
        assertEquals(Arrays.asList("a", "bc", "def", "ghij"), nt.stringData);
    }
    
    public void testCtorBadType() {
        StringBuffer ary[] = {new StringBuffer()};
        try {
            new NativeTensor(ary);
            fail("Shouldn't be able to create a NativeTensor of StringBuffers");
        } catch (IllegalArgumentException e) {
        }
    }
}
