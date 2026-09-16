import single_rgb_train as st

st.MODEL_NAME = 'vit_b_16'
st.RESULT_DIR = 'fusion_result_samm_rgb_single_vit_b_16'
st.TRANSFER_WEIGHTS_PATH = None


def main():
    print("Starting SAMM RGB single-channel experiment: vit_b_16")
    print("Results will be saved to: fusion_result_samm_rgb_single_vit_b_16")
    st.main()


if __name__ == '__main__':
    main()
