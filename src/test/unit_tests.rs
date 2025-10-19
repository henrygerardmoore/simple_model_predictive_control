mod unit_tests {
    use ndarray::Array;

    use crate::mpc_controller::MPCController;

    #[test]
    fn test_input_to_chunks() {
        let input_vec = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 0.];
        let inputs = Array::from_vec(input_vec.clone());
        let view = inputs.view();
        let mut input_chunked = MPCController::<1, 2>::inputs_to_chunks(&view);

        // this should have chunked the input into pairs of twos
        for i in 0_usize..5_usize {
            let cur_val_chunked = input_chunked.next().unwrap();
            assert_eq!(input_vec[2 * i], cur_val_chunked[0]);
            assert_eq!(input_vec[2 * i + 1], cur_val_chunked[1]);
        }

        let input_vec = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let inputs = Array::from_vec(input_vec.clone());
        let view = inputs.view();
        let mut input_chunked = MPCController::<1, 3>::inputs_to_chunks(&view);

        // now this should be chunked into threes
        for i in 0_usize..4_usize {
            let cur_val_chunked = input_chunked.next().unwrap();
            assert_eq!(input_vec[3 * i], cur_val_chunked[0]);
            assert_eq!(input_vec[3 * i + 1], cur_val_chunked[1]);
            assert_eq!(input_vec[3 * i + 2], cur_val_chunked[2]);
        }

        // now let's try with a large array and a large input size
        const LARGE_NUM_STEPS: usize = 100;
        const LARGE_INPUT_SIZE: usize = 23;

        // just make an input vec that size
        let input_vec = (0..(LARGE_NUM_STEPS * LARGE_INPUT_SIZE))
            .map(|index| (index * 3 % 5) as f64 + 3.14 * index as f64)
            .collect::<Vec<f64>>();
        let inputs = Array::from_vec(input_vec.clone());
        let view = inputs.view();

        // test that we have the right number of chunks
        let input_chunked = MPCController::<1, LARGE_INPUT_SIZE>::inputs_to_chunks(&view);
        assert_eq!(input_chunked.count(), LARGE_NUM_STEPS);

        let mut input_chunked = MPCController::<1, LARGE_INPUT_SIZE>::inputs_to_chunks(&view);

        for i in 0..LARGE_NUM_STEPS {
            let cur_val_chunked = input_chunked.next().unwrap();
            for j in 0..LARGE_INPUT_SIZE {
                assert_eq!(input_vec[LARGE_INPUT_SIZE * i + j], cur_val_chunked[j]);
            }
        }
    }
}
