import tensorflow as tf
import collections as pycoll


def tensor_fusion_and_allreduce( g_list, hvd, small_thres=32000, max_group=100 ):
  # print ('==== before fusion gradients list ======')
  # print ( len(g_list) )

  grads, packing = pack_small_tensors( g_list, small_thres, max_group )
  # print ('==== fusion tensors ======')
  # print ( len(grads) )
  # grads_fp16 = [ tf.cast(g, tf.float16) for g in grads ]  #change to fp16 for communiciation

  avg_grads = [ hvd.allreduce(g, average=False) for g in grads ]

  avg_grads_fp32 = [ tf.cast(g, tf.float32) for g in avg_grads ]  #change to fp32 for calculation
  ori_avg_grads = unpack_small_tensors( avg_grads_fp32, packing )

  # print ('==== after fusion gradients list ======')
  # print (len(ori_avg_grads))

  return ori_avg_grads



def pack_small_tensors(g_list, small_thres, max_group):
  # divide all tensors depend on small thres 
  small_indices=[]
  large_indices=[] 
  for idx, g in enumerate( g_list ):
    if g.shape.num_elements() <= small_thres:
      small_indices.append( idx )
    else:
      large_indices.append( idx )
  
  small_ranges, small_singles = extract_ranges( small_indices, range_size_limit=max_group )
  large_indices = sorted( large_indices + small_singles ) 
  
  num_g = len( g_list )
  packing = {}

  if small_ranges:
    new_g_list = []
    
    for r in small_ranges:
      key = '%d' % ( len(new_g_list) )
      new_g = pack_range( r, g_list, key, packing )
      new_g_list.append( new_g )
   
    for i in large_indices:
      new_g_list.append( g_list[i] )
    
    return new_g_list, packing
  else:
    return g_list, None


def unpack_small_tensors( f_list, packing ):
  if not packing:
    return f_list

  num_packed = len( packing.keys() )
  new_g_list = f_list[ num_packed: ]
  
  for i in range( 0, num_packed ):
    k = '%d' % ( i )
    gpt = packing[k]
    new_g = unpack_grad_tuple( f_list[i],gpt )

    for gi, idx in enumerate( gpt.indices ):
      assert idx == gpt.indices[gi]
      new_g_list.insert( idx, new_g[gi] )
  return new_g_list


def unpack_grad_tuple( fused_tensor, gpt ):
  elt_widths = [ x.num_elements() for x in gpt.shapes ]
  splits = tf.split( fused_tensor, elt_widths  )
  unpack_g = []
  for idx, s in enumerate( splits ):
    unpack_g.append( tf.reshape( s, gpt.shapes[idx] ) )
  return unpack_g


GradPackTuple = pycoll.namedtuple('GradPackTuple', 'indices shapes')

def pack_range( rag, g_list, key, packing ):
  to_pack = g_list[rag[0]: rag[1]+1]
  members = []
  restore_shapes = []
  
  for g in to_pack:
    restore_shapes.append( g.shape )
    members.append( tf.reshape(g, [-1]) )
  packing[key] = GradPackTuple( 
                 indices=range( rag[0], rag[1]+1 ),
                 shapes=restore_shapes )

  return tf.concat( members, 0 )



def extract_ranges( index_list, range_size_limit=32 ):
  if not index_list:
    return [], []

  first = index_list[0]
  last = first
  ranges = []
  singles = []
  for i in index_list[1:]:
    if i == last+1 and ( last-first ) <= range_size_limit:  # ensure consecutive
      last = i 
    else:
      if last > first:
        ranges.append( [first, last] )
      else:
        singles.append(first)
      first = i
      last = i
  if last > first:
    ranges.append( [first, last] )
  else:
    singles.append(first)
  return ranges, singles



