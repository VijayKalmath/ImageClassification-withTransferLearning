	[&??|0B@[&??|0B@![&??|0B@	???j7X1@???j7X1@!???j7X1@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6[&??|0B@r?z?fa'@1Dl?p?? @AB?????@I<?b?ϊ@Y??~31=@*	?C?l???@2F
Iterator::Model?@?"?"@!?k[1F@)K9_콈@1??(aB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??&G@!92????A@)~8gD?@1?C?b?g6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ܵ??@!A8Y??*@)??ܵ??@1A8Y??*@:Preprocessing2U
Iterator::Model::ParallelMapV2-????!?^??S @)-????1?^??S @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice4?i??r??!?Tq??@)4?i??r??1?Tq??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?rg&?'@!B?????K@)?????+??1??'UF@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?\S ?S	@!??6,??-@)?2?????1F-?c? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??<??@!&?gz$?$@):?`?????1?e???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 17.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?20.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t32.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???j7X1@I?A??M@Q??Z?$7@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	r?z?fa'@r?z?fa'@!r?z?fa'@      ??!       "	Dl?p?? @Dl?p?? @!Dl?p?? @*      ??!       2	B?????@B?????@!B?????@:	<?b?ϊ@<?b?ϊ@!<?b?ϊ@B      ??!       J	??~31=@??~31=@!??~31=@R      ??!       Z	??~31=@??~31=@!??~31=@b      ??!       JGPUY???j7X1@b q?A??M@y??Z?$7@