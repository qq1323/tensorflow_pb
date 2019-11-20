# 版权声明：本文为CSDN博主「millions_luo」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/luoyanjunhehehe/article/details/90902804
# 转化后Value Error问题

# Value Error错误信息一般为
# ValueError: Input 0 of node … was passed float from … incompatible with expected float_ref.
# 对其进行类型转换即可

from tensorflow.python.tools import freeze_graph

def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    '''
    # 输出节点名称
    output_node_names = "out/pred" 
    
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        
        # fix batch norm nodes
        # 解决value error问题
        for node in input_graph_def.node:
          if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
              if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
          elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

        output_graph_def = graph_util.convert_variables_to_constants(  
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))
            
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node)) 

if __name__ == '__main__':
	input_checkpoint = 'models/ckpt/epoch_50.ckpt'
	output_graph = 'freeze_model.pb'
	freeze_graph(input_checkpoint, output_graph)

