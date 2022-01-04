    # # Init master figure    
    # fig = plt.figure(constrained_layout=True, figsize=(len(layers) + 1, len(factors) + 1))
    # gs = fig.add_gridspec(nrows=len(factors) + 1, ncols=len(layers) + 1)
    # ax = fig.add_subplot(gs[0, 0])
    # ax.axis('off')
    
            #rdm_N = cka.get_rdm(a_N).numpy()
        # ax = fig.add_subplot(gs[0, i+1])
        # ax.imshow(rdm_N)
        # title = '\n'.join([s for s in layer.name.split('_') if 'block' not in s])
        # if 'mean' in title:
        #     title = 'global\npool'

        # ax.set_title(title)
        # ax.axis('off')
#         key = v if isinstance(v, str) else '_'.join(v)
    #         rs[key].append(r)

    #         print(f"Layer {layer.name}, feature {v}: {r:.3f}")
    #         ax = fig.add_subplot(gs[ii + 1, 0])
    #         rdm_F = cka.get_rdm(a_F)
    #         ax.imshow(rdm_F)
    #         ax.axis('off')
    #         title = F_NAMES[key]
    #         ax.set_title(title, fontsize=12)

    # for i, (name, r) in enumerate(rs.items()):        
    #     ax = fig.add_subplot(gs[i+1, 1:])
    #     ax.bar(np.arange(len(layers)), r, width=0.45)
    #     ax.set_ylim(0, 1)
    #     ax.set_xlim(-.25, len(layers) - .75)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.set_ylabel('CKA', fontsize=12)
    #     ax.yaxis.set_label_position("right")
    #     ax.yaxis.tick_right()

    # fig.savefig(f'figures/{model_name}.png', dpi=300, bbox_inches='tight')