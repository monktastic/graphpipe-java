// automatically generated by the FlatBuffers compiler, do not modify

package com.oracle.graphpipefb;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class MetadataRequest extends Table {
  public static MetadataRequest getRootAsMetadataRequest(ByteBuffer _bb) { return getRootAsMetadataRequest(_bb, new MetadataRequest()); }
  public static MetadataRequest getRootAsMetadataRequest(ByteBuffer _bb, MetadataRequest obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; }
  public MetadataRequest __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }


  public static void startMetadataRequest(FlatBufferBuilder builder) { builder.startObject(0); }
  public static int endMetadataRequest(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
}

