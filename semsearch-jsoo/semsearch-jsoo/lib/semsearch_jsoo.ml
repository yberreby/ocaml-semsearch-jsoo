let calculate_similarities_with_base base_tensor encoded_sentences :
    (int * float) Array.t =
  let similarities_tensor =
    Tfjs.matmul base_tensor (Tfjs.transpose encoded_sentences)
  in
  let similarities_values = Tfjs.Tensor.to_array similarities_tensor in
  let arr =
    Array.init (Array.length similarities_values) (fun i ->
        (i, similarities_values.(i)))
  in
  Array.sort
    (fun (_, similarity1) (_, similarity2) ->
      -Float.compare similarity1 similarity2)
    arr;
  arr
