import tensorflow as tf
from tensorflow.python.ops import variable_scope
from static_utils.stack_utils import npu_unstack, npu_stack
def static_decoder(decoder, 
                   maximum_iterations, 
                   output_time_major, 
                   swap_memory, 
                   scope
                  ):
    all_outputs_rnn_output = []
    all_outputs_sample_id = []
    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # initialize decoder 
        initial_finished, initial_inputs, initial_state = decoder.initialize()

        inputs = initial_inputs
        state = initial_state
        for i in range(maximum_iterations):
            if i > 0:
                varscope.reuse_variables()
            time = tf.convert_to_tensor(i)
            call_decoder = lambda:decoder.step(time, inputs, state)
            (output, state, inputs, decoder_finished) = call_decoder()
            #if i == 0:
            all_outputs_rnn_output.append(output.rnn_output)
            all_outputs_sample_id.append(output.sample_id)


        final_rnn_output = npu_stack(all_outputs_rnn_output, axis=0)
        final_sample_id = npu_stack(all_outputs_sample_id, axis=0)
    #    output.rnn_output = final_rnn_output
    #    output.sample_id = final_sample_id
           
    return (final_rnn_output, final_sample_id), state    
    exit(0)

