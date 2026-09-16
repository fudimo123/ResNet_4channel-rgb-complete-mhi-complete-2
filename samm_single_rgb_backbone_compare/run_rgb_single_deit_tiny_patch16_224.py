import single_rgb_train as st

st.MODEL_NAME = 'deit_tiny_patch16_224'
st.RESULT_DIR = 'fusion_result_samm_rgb_single_deit_tiny_patch16_224'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting SAMM RGB single-channel experiment: deit_tiny_patch16_224")
    print("Results will be saved to: fusion_result_samm_rgb_single_deit_tiny_patch16_224")
    st.main()


if __name__ == '__main__':
    main()
