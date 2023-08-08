let load_local_model json_path =
  let handler = json_path |> Tfjs.IO.fileSystem in
  Tfjs.load_graph_model handler

let calculate_similarities_with_base base_tensor encoded_sentences =
  let similarities_tensor =
    Tfjs.matmul base_tensor (Tfjs.transpose encoded_sentences)
  in
  let similarities_values = Tfjs.Tensor.to_array similarities_tensor in
  Array.init (Array.length similarities_values) (fun i ->
      (i, similarities_values.(i)))
  |> Array.to_list
  |> List.sort (fun (_, similarity1) (_, similarity2) ->
         -Float.compare similarity1 similarity2)
