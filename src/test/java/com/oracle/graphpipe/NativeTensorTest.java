package com.oracle.graphpipe;

import com.oracle.graphpipefb.Tensor;
import com.oracle.graphpipefb.Type;
import junit.framework.TestCase;
import org.junit.Assert;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class NativeTensorTest extends TestCase {
    private byte[] fromByteBuffer(ByteBuffer bb) {
        byte[] data = new byte[bb.remaining()];
        bb.get(data);
        return data;
    }

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

    short rank3Ary[][][] = {
            {{0x0001, 0x0020}, {0x0300, 0x4000}},
            {{0x000A, 0x00B0}, {0x0C00, (short)0xD000}}
    };
    // rank3Ary in little-endian order.
    byte[] rank3AryData = {
            0x01, 0x00, 0x20, 0x00, 0x00, 0x03, 0x00, 0x40,
            0x0A, 0x00, (byte)0xB0, 0x00, 0x00, 0x0C, 0x00, (byte)0xD0,
    };
    short rank3AryFlat[] = {
            0x0001, 0x0020, 0x0300, 0x4000,
            0x000A, 0x00B0, 0x0C00, (short)0xD000
    };

    public void testCtorRank3() {
        NativeTensor nt = new NativeTensor(rank3Ary);
        Assert.assertArrayEquals(rank3AryData, nt.data.array());
    }
    
    public void testCtorFlat() {
        long[] shape = {rank3Ary.length, rank3Ary[0].length, rank3Ary[0][0].length};
        Tensor t1 = new NativeTensor(rank3AryFlat, shape).toTensor();
        Tensor t2 = new NativeTensor(rank3Ary).toTensor();
        assertNumericTensorsEqual(t1, t2);
    }
    
    private void assertNumericTensorsEqual(Tensor t1, Tensor t2) {
        assertEquals(t1.type(), t2.type());
        Assert.assertArrayEquals(
                t1.shapeAsByteBuffer().array(), 
                t2.shapeAsByteBuffer().array());
        Assert.assertArrayEquals(
                t1.dataAsByteBuffer().array(),
                t2.dataAsByteBuffer().array());
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
   
    public void testToTensor_Numeric() {
        NativeTensor nt = new NativeTensor(rank3Ary);
        Tensor t = nt.toTensor();

        // Fetch the byte data.
        byte[] data = fromByteBuffer(t.dataAsByteBuffer());
        Assert.assertArrayEquals(rank3AryData, data);
     
        // Compare shapes.
        assertEquals(rank3Ary.length, t.shape(0));
        assertEquals(rank3Ary[0].length, t.shape(1));
        assertEquals(rank3Ary[0][0].length, t.shape(2));
       
        // Short is type 4.
        assertEquals(4, t.type());
    }

    public void testToTensor_String() {
        String ary[][] = {{"a", "bc"}, {"def", "ghij"}};
        NativeTensor nt = new NativeTensor(ary);
        Tensor t = nt.toTensor();

        // Compare strings.
        assertEquals(4, t.stringValLength());
        assertEquals(ary[0][0], t.stringVal(0));
        assertEquals(ary[0][1], t.stringVal(1));
        assertEquals(ary[1][0], t.stringVal(2));
        assertEquals(ary[1][1], t.stringVal(3));

        // Compare shapes.
        assertEquals(ary.length, t.shape(0));
        assertEquals(ary[0].length, t.shape(1));

        // Compare type.
        assertEquals(Type.String, t.type());
    }
    
    public void testFromTensor() {
        double ary[][][] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        NativeTensor nt = new NativeTensor(ary);
        NativeTensor nt2 = new NativeTensor(nt.toTensor());
        
        ByteBuffer bb = ByteBuffer.allocate(8 * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 1; i <= 8; i++) {
            bb.putDouble(i);
        }
        bb.rewind();
        Assert.assertArrayEquals(fromByteBuffer(bb), fromByteBuffer(nt2.data));
        assertEquals(Arrays.asList(2L, 2L, 2L), nt2.shape);
        assertEquals(Double.class, nt2.numConv.clazz);
    }
    
    public void testToINDArray() {
        double ary[][][] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        NativeTensor nt = new NativeTensor(ary);
        INDArray ndArr = nt.toINDArray();

        for (int i = 0; i < ary.length; i++) {
            for (int j = 0; j < ary[i].length; j++) {
                for (int k = 0; k < ary[i][j].length; k++) {
                    assertEquals(ary[i][j][k], ndArr.getDouble(i, j, k));
                    // Weirdly, the syntax for Float is:
                    // ndArr.getFloat(new int[]{i, j, k});
                }
            }
        }
        
    }

    public void testFromINDArray() {
        double ary[][][] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        NativeTensor nt = new NativeTensor(ary);
        NativeTensor nt2 = new NativeTensor(nt.toINDArray());

        // INDArray is (currently?) stored as floats even if created with
        // Doubles.
        ByteBuffer bb = ByteBuffer.allocate(4 * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 1; i <= 8; i++) {
            bb.putFloat(i);
        }
        bb.rewind();
        assertEquals(Float.class, nt2.numConv.clazz);
        Assert.assertArrayEquals(fromByteBuffer(bb), fromByteBuffer(nt2.data));
        assertEquals(Arrays.asList(2L, 2L, 2L), nt2.shape);
    }

    public void testToFlatArray_Numeric() {
        double ary[][][] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
        NativeTensor nt = new NativeTensor(ary);

        double[] ary2 = (double[])nt.toFlatArray();
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, ary2, 0);
    }

    public void testToFlatArray_String() {
        String ary[][] = {{"a", "bc"}, {"def", "ghij"}};
        NativeTensor nt = new NativeTensor(ary);

        String[] ary2 = (String[])nt.toFlatArray();
        Assert.assertArrayEquals(new String[]{"a", "bc", "def", "ghij"}, ary2);
    }
}
