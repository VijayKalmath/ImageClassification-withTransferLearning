		?"hLKw@	?"hLKw@!	?"hLKw@	}}b?:@}}b?:@!}}b?:@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6	?"hLKw@_}<??bc@1X S?#@A?wak?"?@I??(S@Yc?ZB>Y@*	?Mb>?A2F
Iterator::Model p??s?b@!;ݱM@)?????.`@1ܛ?I??H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?qS?F@!-???1@),???d?@1? oiQ(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?X???9@!???]?#@)?X???9@1???]?#@:Preprocessing2U
Iterator::Model::ParallelMapV2?
?ro5@!??9MV? @)?
?ro5@1??9MV? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(?NlK@!?[??,5@)rP?LK/@1?R[ 1)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?k]j??,@!?2֯a7@)?k]j??,@1?2֯a7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&???{?C@!?;??C.@)˟o?*@14?m?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?'??&
[@!???"N?D@)??-Y?@1??B???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 26.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?20.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t41.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9}}b?:@IPx??ˡQ@Q+
??a@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	_}<??bc@_}<??bc@!_}<??bc@      ??!       "	X S?#@X S?#@!X S?#@*      ??!       2	?wak?"?@?wak?"?@!?wak?"?@:	??(S@??(S@!??(S@B      ??!       J	c?ZB>Y@c?ZB>Y@!c?ZB>Y@R      ??!       Z	c?ZB>Y@c?ZB>Y@!c?ZB>Y@b      ??!       JGPUY}}b?:@b qPx??ˡQ@y+
??a@