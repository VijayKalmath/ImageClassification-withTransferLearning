?	=Y???@@=Y???@@!=Y???@@	???u??@???u??@!???u??@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6=Y???@@ӡ??n?!@1˂????!@AnQf?Lr@IS???.)$@Y?~?٭E@*	???S??@2F
Iterator::Model?I?%r9"@!JcO??H@)c`??@1l?h??]C@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensort]???T@!?X
?T.@)t]???T@1?X
?T.@:Preprocessing2U
Iterator::Model::ParallelMapV2??DR???!x?WQ?%@)??DR???1x?WQ?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceӿ$?)&??!B?"??#@)ӿ$?)&??1B?"??#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?T?t<@!??n?xz8@)???O?n??1W?ߠ"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?QH2??"@!?????I@)=a????1iH?N?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??e?i?@![??v?3@)?vN?@;??1? ?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????ɭ@!Dv?XZ,@)_F???j??1(??4?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?30.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t26.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???u??@I?a?N?_P@Q???Ʉ:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ӡ??n?!@ӡ??n?!@!ӡ??n?!@      ??!       "	˂????!@˂????!@!˂????!@*      ??!       2	nQf?Lr@nQf?Lr@!nQf?Lr@:	S???.)$@S???.)$@!S???.)$@B      ??!       J	?~?٭E@?~?٭E@!?~?٭E@R      ??!       Z	?~?٭E@?~?٭E@!?~?٭E@b      ??!       JGPUY???u??@b q?a?N?_P@y???Ʉ:@?"D
"sequential/vgg16/block4_conv2/Relu_FusedConv2D?i??a??!?i??a??"D
"sequential/vgg16/block5_conv3/Relu_FusedConv2D??ie????!?~????"D
"sequential/vgg16/block4_conv3/Relu_FusedConv2D$?ヘʸ?!(0*????"D
"sequential/vgg16/block5_conv1/Relu_FusedConv2D?31ĸ?!.4hw????"D
"sequential/vgg16/block5_conv2/Relu_FusedConv2Dp?X?????!
a??l??"D
"sequential/vgg16/block1_conv2/Relu_FusedConv2D?lR????!~?u5???"D
"sequential/vgg16/block2_conv2/Relu_FusedConv2D???Sݭ?!7?Z?
y??"D
"sequential/vgg16/block4_conv1/Relu_FusedConv2D	#Rfo
??!hܿ??)??"D
"sequential/vgg16/block3_conv3/Relu_FusedConv2D????6_??!G?????"D
"sequential/vgg16/block3_conv2/Relu_FusedConv2D{??n???![? ?s??Q      Y@Y       @a      W@q¿??xP@y???ޑ??"?
both?Your program is MODERATELY input-bound because 8.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?30.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t26.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?65.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 