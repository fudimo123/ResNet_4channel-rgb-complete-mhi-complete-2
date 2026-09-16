import single_rgb_train as st

st.MODEL_NAME = 'shufflenet_v2_x1_0'
st.RESULT_DIR = 'fusion_result_samm_rgb_single_shufflenet_v2_x1_0'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting SAMM RGB single-channel experiment: shufflenet_v2_x1_0")
    print("Results will be saved to: fusion_result_samm_rgb_single_shufflenet_v2_x1_0")
    st.main()


if __name__ == '__main__':
    main()
