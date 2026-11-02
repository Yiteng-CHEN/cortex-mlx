const $=id=>document.getElementById(id);
// ======== 中 / 英 语言切换：界面文字按短语替换（支持带数字的动态文字）；聊天内容（你的提问和模型的回答）不翻译 ========
const EN=[
["Neurolume 大脑","Neurolume Brain"],
["三维全息神经网络，每一圈节点是一层，颜色表示该层的实时活动","3D holographic neural network: each ring of nodes is a layer, colour shows its live activity"],
["切换大脑的形态：星云 / 球体","Switch the brain shape: nebula / sphere"],
["形态：星云","Shape: Nebula"],["形态：球体","Shape: Sphere"],
["高","High"],["低","Low"],
["聊天","Chat"],["选择模型","Choose model"],["适配器","Adapter"],["模型库","Models"],["模型","Model"],["训练","Train"],
["打开后，模型需要时会自己上网搜索并读网页（你的问题会发给搜索服务）","When on, the model can search the web and read pages by itself (your question is sent to the search service)"],
["联网搜索：开","Web search: on"],["联网搜索","Web search"],["搜索设置","Search settings"],
["导入书和文档，对话时模型会先从里面查资料再回答","Import books and documents; the model looks things up in them before answering"],
["知识库：开","Knowledge base: on"],["知识库","Knowledge base"],
["清空对话","Clear chat"],
["问点什么，Enter 发送，Shift+Enter 换行","Ask something. Enter to send, Shift+Enter for a new line"],
["消息","Message"],["发送","Send"],["神经网络层","Neural network layers"],
["层，层与层之间的连线随活动发光"," layers; links between layers glow with activity"],
["显示模式","Display mode"],["推理","Inference"],["权重","Weights"],["训练一步（示例）","Train one step (sample)"],["关闭","Close"],
["Devstral Small 2 · 40 层 · 全部为标准注意力","Devstral Small 2 · 40 layers · all standard attention"],
["Qwen3.6-27B · 64 层 = 16 ×（3 线性注意力 + 1 全注意力）","Qwen3.6-27B · 64 layers = 16 × (3 linear attention + 1 full attention)"],
["注意力 Q","Attention Q"],["注意力 输出","Attention out"],["MLP 门控","MLP gate"],["MLP 下投影","MLP down-proj"],
["演示后端 · 40 层","Demo backend · 40 layers"],["演示后端 · 模拟","Demo backend · simulated"],
["权重值：","Weight values: "],["真实权重","Real weights"],["示例（运行 dump_weights.py 换成真实值）","Sample (run dump_weights.py for real values)"],
["硬件占用：读取中…","Hardware: reading…"],["硬件占用：模拟","Hardware: simulated"],["球体 = ","Sphere = "],["显卡 ","GPU "],["内存 ","Memory "],["· 模型 ","· model "],
["位置：行 ","Position: row "],["，列 ",", col "],["权重：","Weight: "],["（示例）"," (sample)"],["（模拟）"," (simulated)"],
["该矩阵标准差：","Matrix std dev: "],["该层当前活动：","Layer activity now: "],
["真实，来自适配器","real, from the adapter"],["示例","sample"],
["高：该层这一步对当前 token 的改动幅度","High: how much this layer changed the current token"],
["连线发光 = 相邻两层同时活跃，亮点沿线移动","Glowing link = adjacent layers active together; bright dots travel along it"],
["权重增大","Weight increased"],["权重减小","Weight decreased"],
["ΔW 来自适配器的真实值（与权重图同一位置）","ΔW is the adapter's real value (same positions as the weight map)"],
["ΔW 为示例；训练后会换成适配器的真实值","ΔW is a sample; after training it becomes the adapter's real value"],
["正权重","Positive weight"],["负权重","Negative weight"],
["颜色深浅 = |权重| ÷ 该矩阵标准差（3σ 封顶）","Colour depth = |weight| ÷ matrix std dev (capped at 3σ)"],
["实时 · 真实模型","Live · real model"],["信号：","Signal: "],["来自模型的实时逐层活动","live per-layer activity from the model"],["模拟","simulated"],
["未加载模型","No model loaded"],["等待模型","waiting for a model"],["演示模式 · 未连接本地服务","Demo mode · local service not connected"],
["这是演示回复：页面没有连接到本地服务，所以回答和各层的活动都是模拟的。","This is a demo reply: the page is not connected to the local service, so the answer and layer activity are simulated."],
["在 Mac 上运行","On your Mac run"],
["后打开 http://127.0.0.1:8765，这里就会显示 Devstral 真实生成的文字，大脑图和网络层会用它真实的逐层活动着色。","then open http://127.0.0.1:8765 — you will see Devstral's real output, and the brain and layer views will be coloured by its real per-layer activity."],
["来源：","Sources: "],["🌐 已联网搜索","🌐 Searched the web"],
["提示：这次回复没有收到逐层活动信号，所以立柱和层图不会亮。请把终端窗口里最后 20 行发给开发者。","Note: this reply carried no per-layer activity signal, so the brain and layer views will not light up. Please send the last 20 lines of the terminal window to the developer."],
["出错：","Error: "],
["联网搜索需要先启动本地服务。","Web search needs the local service to be running."],
["已开启联网搜索：模型需要时会自己搜索，你的问题会发给搜索服务。","Web search on: the model searches when it needs to; your question is sent to the search service."],
["已关闭联网搜索：Neurolume 不会向外发送任何内容。","Web search off: Neurolume sends nothing out."],
["联网搜索打开后，模型需要最新信息时会自己搜索并阅读网页，回答里会标出来源。默认关闭；搜索时你的问题会发给所选搜索服务。网页内容只当作资料，不会当作指令执行；不会读取本机或内网地址。","When web search is on, the model searches and reads pages by itself when it needs fresh information, and cites its sources. Off by default; when searching, your question is sent to the chosen search service. Page content is treated as reference data, never as instructions, and local or internal addresses are never read."],
["DuckDuckGo（免费，不需要账号）","DuckDuckGo (free, no account)"],
["我自己的 SearXNG（更稳定、更隐私，需要自己部署）","My own SearXNG (more stable and private, you host it)"],
["SearXNG 地址，如 http://localhost:8080","SearXNG address, e.g. http://localhost:8080"],["SearXNG 地址","SearXNG address"],
["保存","Save"],
// ---- 回答时间轴 / 逐层预测 / 训练回看
["选中的内容","Selection"],["最近一次回答","Latest answer"],["正在生成","Generating"],[" · 拖动或点击热力图回看任意一个字"," · drag or click the heatmap to revisit any token"],
["逐层预测显示","Logit lens shows"],["逐层预测显示哪些层","Which layers the logit lens shows"],["全部层","all layers"],["每 2 层","every 2nd layer"],["每 4 层","every 4th layer"],
["取消选中，回到最近一次回答的全部内容","Clear the selection and show the whole latest answer"],["看最近一次回答","Latest answer"],
["整段回答的逐字逐层活动热力图：横向是每个字，纵向是每一层；点击或拖动回看","Per-token, per-layer activity of the whole answer: tokens across, layers down; click or drag to revisit"],
["选择第几个字","Choose a token"],["横向：回答里的每个字 · 纵向：第 0 层（上）到最后一层（下） · 在对话里选中一段 AI 回答，就只看那一段","Across: every token of the answer · Down: layer 0 (top) to the last layer (bottom) · Select part of an AI answer in the chat to see only that part"],
["定下来","Settled"],["已等于最终输出","already equals the final output"],
["打开这个视图后再提问，就能看到每一层在“猜”下一个字。","Ask something with this view open to see every layer \"guess\" the next token."],
["还没有训练过的适配器：层图里现在是示例 ΔW。训练完成后，这里会显示真实的各层改变幅度，并且可以随时回来查看。","No trained adapter yet: the layer map shows a sample ΔW. After training, this shows the real per-layer change, and you can come back to it any time."],
["查看适配器 ","Viewing adapter "],["选择要查看的适配器","Choose an adapter to view"],[" · 验证损失 "," · validation loss "],
[" · 每根柱子是一层被训练改变的总幅度（ΔW 大小），悬停层图看具体位置和数值"," · each bar is how much training changed a layer (size of ΔW); hover the layer map for positions and values"],

// ---- 工作室：信号、逐层预测、回放、对比、导出、数据生成、图解
["每生成一个字，看看每一层“如果在这里停下”会预测什么（logit lens，需要真实模型）","For every generated token, see what each layer would predict \"if the model stopped here\" (logit lens; needs a real model)"],
["逐层预测","Logit lens"],["显示：增强","View: enhanced"],["显示：原始","View: raw"],
["增强：放大层与层之间的差别，便于观察；原始：直接按测量值的比例显示","Enhanced: exaggerates differences between layers so they are easy to see. Raw: shows measured values in proportion"],
["这张图怎么看？","How to read this?"],["同一个问题，看原模型和训练后的回答有什么不同","Ask the same question and see how the original and the trained model differ"],["对比原模型","Compare with base"],
["每一列是一个生成的字，每一行是一层：格子里是这一层“如果在这里停下”最可能输出的字，颜色越深越有把握。最下面一行是模型真正输出的字。","Each column is a generated token and each row a layer: a cell shows the token that layer would most likely output \"if the model stopped here\" — darker means more confident. The bottom row is what the model actually output."],
["Q（查询）：这一层的每个字“想找什么信息”","Q (query): what each token in this layer is \"looking for\""],["输出：把注意力找到的信息写回这个字的表示里","Output: writes what attention found back into the token's representation"],
["门控：决定 MLP 里哪些“知识通道”打开","Gate: decides which \"knowledge channels\" in the MLP open"],["下投影：把 MLP 算出的结果压回原来的维度，写回去","Down-projection: compresses the MLP's result back to the model width and writes it back"],
["测量值：","Measured: "],["逐层预测需要连接真实模型。","The logit lens needs a real model."],
["打开这个视图后再提问，就能看到每一层在“猜”下一个字。（会让生成稍慢一些）","With this view open, ask something and watch each layer \"guess\" the next token. (Generation gets a little slower.)"],
["输出","Output"],["▶ 回放","▶ Replay"],["❚❚ 暂停","❚❚ Pause"],["回放位置","Replay position"],
["训练完成","Training complete"],["训练损失 ","Training loss "],["验证集损失：训练前 ","Validation loss: before "],[" → 训练后 "," → after "],["（降低 "," (down "],["（升高 "," (up "],
["验证集是没参与训练的那部分数据，损失越低说明模型越会“举一反三”。","The validation set is data that was not used for training; lower loss means the model generalises better."],
["在没参与训练的验证数据上明显变好了。","It clearly improved on validation data it never trained on."],
["在验证数据上略有改善，可以再多准备些数据或用“标准”强度。","Slight improvement on validation data; try more data or the \"Standard\" intensity."],
["在验证数据上没有变好，可能是数据太少或和验证数据差别大，建议先对比看看回答。","No improvement on validation data — maybe too little data or it differs from the validation data. Compare the answers first."],
["各层改变了多少","How much each layer changed"],
["每根柱子是一层，高度是这一层被训练改变的幅度（适配器 ΔW 的大小）。主页的“训练”视图能看到具体位置。","Each bar is a layer; its height is how much training changed it (the size of the adapter's ΔW). The \"Train\" view on the home page shows where."],
["建议先用“试用并对比”看看同一个问题训练前后的回答，再决定是否继续使用。效果不好就点“恢复原模型”，不会有任何损失。","Use \"Try & compare\" to see answers to the same question before and after training, then decide. If it's not good, click \"Restore original model\" — nothing is lost."],
["试用并对比","Try & compare"],["导出…","Export…"],["导出这个模型…","Export this model…"],
["效果对比","Compare"],["先在上方的适配器下拉框里选一个训练结果，再来对比。","First pick a training result in the adapter menu above, then compare."],
["同一个问题分别让原模型和训练后的模型回答（不重新加载模型，适配器临时关掉再打开）。用没参与训练的验证题最能看出真实效果。","The same question is answered by the original and the trained model (no reload: the adapter is switched off and on). Validation questions it never trained on show the real effect best."],
["验证集里的问题（点一下填入）","Questions from the validation set (click to use)"],["输入一个问题","Type a question"],["对比用的问题","Question to compare"],["对比","Compare"],
["正在生成两次回答…","Generating both answers…"],["数据里的参考答案","Reference answer from the data"],["原模型","Original model"],["训练后","After training"],
["导出模型","Export model"],["把训练好的适配器合并进模型，得到一个可以单独使用的新模型。导出时会先卸载当前模型，需要几分钟和较多磁盘空间。","Merge the trained adapter into the model to get a standalone model. The current model is unloaded first; this takes a few minutes and quite a lot of disk space."],
["适配器：","Adapter: "],["名字（可选）","Name (optional)"],["例如 my-coder","e.g. my-coder"],["导出名字","Export name"],
["导出为 MLX 模型（LM Studio 可直接用）","Export as an MLX model (works in LM Studio)"],["导出到 Ollama","Export to Ollama"],
["导出到 Ollama 会先把权重还原成未量化的格式（体积约为原来的 4 倍），再由 Ollama 导入并压缩。新结构的模型 Ollama 不一定支持，失败时导出的文件夹会保留。","Exporting to Ollama first restores unquantized weights (about 4× the size), then Ollama imports and compresses them. Ollama may not support newer architectures; the exported folder is kept if it fails."],
["正在导出…","Exporting…"],["导出完成","Export complete"],["导出失败","Export failed"],["已放进 LM Studio 的模型目录：","Added to LM Studio's models folder: "],
["Ollama 里可以直接运行：","Run it in Ollama: "],["没有找到 Ollama。安装后在终端运行：","Ollama was not found. After installing it, run in Terminal: "],["已导出的模型","Exported models"],
["没有现成的训练数据？","No training data yet?"],["选几份你自己的资料（文档、笔记、代码），让当前加载的模型帮你生成问答对，你检查修改后再拿去训练。","Pick a few of your own files (docs, notes, code) and let the loaded model generate Q&A pairs; review and edit them, then train."],
["用我的资料生成训练数据…","Generate training data from my files…"],
["选几份资料，当前加载的模型会帮你生成训练数据。生成的每一条都要你检查：不对的取消勾选或直接改，然后保存成数据集。","Pick a few files and the loaded model will generate training data. Review every item: untick or edit the wrong ones, then save them as a dataset."],
["学会这些知识","Learn this knowledge"],["从文档里生成“问题 + 答案”","Generate question + answer pairs from documents"],["学会我的风格","Learn my style"],
["从你写的代码或文章里，反推“需求”，答案就是你自己写的原文","Infer the request behind your own code or writing; the answer is your original text"],
["选择资料","Choose files"],["开始生成","Start"],["返回","Back"],["请先选择文件","Choose files first"],["停止","Stop"],
["资料较多，这次只处理了前 40 段。","Lots of material: only the first 40 pieces were processed this time."],["取消勾选不要的，可以直接修改文字。","Untick what you don't want; you can edit the text directly."],
["问题","Question"],["答案","Answer"],["数据集名字","Dataset name"],["保存为数据集并继续","Save as dataset and continue"],
["你在看什么","What you are looking at"],["模型由很多“层”叠起来，每生成一个字，信息都会从第一层流到最后一层。层图里每一列是一层，每一行是层里的一个矩阵；脑图里每一圈（或星云的每一段高度）也对应一层。","A model is a stack of layers; for every token, information flows from the first layer to the last. In the layer view each column is a layer and each row a matrix inside it; in the brain view each ring (or each height of the nebula) is one layer too."],
["亮度测的是什么","What the brightness measures"],["每生成一个字，测量每一层对这个字的内部表示改了多少：改动的平均大小除以输入的平均大小。这是真实测量值，不代表“重要性”。“增强”显示会放大层与层之间的差别，“原始”显示直接按测量值的比例。","For each token we measure how much each layer changed its internal representation: the mean size of the change divided by the mean size of the input. It is a real measurement, not a claim about \"importance\". \"Enhanced\" exaggerates differences between layers; \"raw\" shows the values in proportion."],
["注意力（Q、输出）","Attention (Q, output)"],["让当前的字去“看”前面的字：Q 决定要找什么信息，输出矩阵把找到的信息写回来。全注意力层每一步都能看到前面所有字。","Lets the current token look at earlier tokens: Q decides what to look for and the output matrix writes what was found back. Full-attention layers see every earlier token at each step."],
["线性注意力层","Linear-attention layers"],["Qwen3.6 里四层中有三层是线性注意力（DeltaNet）：不再逐个回看前面的字，而是维护一份不断更新的“记忆”，长文本时更快。层号旁边的 ◆ 表示全注意力层。","In Qwen3.6 three of every four layers use linear attention (DeltaNet): instead of looking back at each earlier token they keep a continuously updated \"memory\", which is faster on long text. A ◆ next to a layer number marks full attention."],
["MLP（门控、下投影）","MLP (gate, down-projection)"],["对每个字单独做的计算，很多“知识”存放在这里：门控决定打开哪些通道，下投影把结果写回去。","Per-token computation where much of the \"knowledge\" lives: the gate decides which channels open and the down-projection writes the result back."],
["训练时改了什么（LoRA）","What training changes (LoRA)"],["微调不会改动原模型文件，而是给部分矩阵加上一个很小的修正 ΔW = B×A。“训练”视图显示 ΔW 在哪里、是变大还是变小；训练完成面板里的柱子是每层改动的总量。","Fine-tuning doesn't touch the original model files; it adds a small correction ΔW = B×A to some matrices. The \"Train\" view shows where ΔW is and whether weights grow or shrink; the bars in the training-complete panel show each layer's total change."],
["逐层预测（logit lens）","Logit lens"],["把某一层的中间结果直接送到模型的输出端，看看“如果在这一层停下”会输出什么字。前面的层还没有形成能直接“读出来”的表示，常出现乱码、标点或别的语言，这是正常现象；越往后越接近最终答案。表格最下面的“定下来”一行，是从第几层起每一层都猜中了最终输出：数字越小，这个字越早被决定。下方的时间轴可以拖动回看整段回答里任意一个字。","Send a layer's intermediate result straight to the output to see what token it would produce \"if the model stopped here\". Early layers have not formed a readable representation yet, so gibberish, punctuation or other languages are normal; later layers converge on the final answer. The \"Settled\" row at the bottom is the layer from which every layer already predicts the final output: the smaller the number, the earlier that token was decided. Drag the timeline below to revisit any token of the answer."],
["更详细的说明见项目里的 docs/SIGNALS.md。","More detail in docs/SIGNALS.md in the project."],
// ---- 服务端：工作室
["请先在上方选择一个适配器（训练结果），才能和原模型对比","Choose an adapter (training result) above first to compare with the original model"],["问题是空的","The question is empty"],
["正在训练，请等训练结束再导出","Training is running; wait until it finishes before exporting"],["已有导出在进行","An export is already running"],["同名导出已存在，请换个名字或先删除","An export with this name exists; choose another name or delete it first"],
["磁盘空间不够：导出大约需要 ","Not enough disk space: the export needs about "],[" GB，可用 "," GB, available "],
["Ollama 导入失败（可能还不支持这个模型结构）：","Ollama import failed (it may not support this architecture yet): "],
["生成训练数据需要先加载一个真实模型","Generating training data needs a real model loaded first"],["正在生成训练数据，请稍候","Training data is being generated; please wait"],
["没有读到可用的内容（文件太短或格式不支持）","No usable content (files too short or unsupported)"],

["让模型读取你选的项目文件夹并提出修改（每次修改都要你批准）","Let the model read the project folder you choose and propose edits (every edit needs your approval)"],
["已开启 Agent，但还没有选择项目文件夹：请先点“Agent”按钮选择。","Agent is on but no project folder is chosen yet: click the “Agent” button to choose one."],
["选一个项目文件夹后，模型可以查看里面的文件、搜索代码，并提出修改。每一处修改都会先显示 diff，等你点“应用修改”才会真正写入；写入后可以在下面撤销。密钥、.env 和 .git 里的文件不会被读取或修改。","After you choose a project folder, the model can look at its files, search the code and propose edits. Every edit is shown as a diff first and is written only when you click “Apply”; after that you can undo it below. Secrets, .env and files in .git are never read or changed."],
["聊天时启用 Agent（读取项目并提出修改）","Use Agent in chat (read the project and propose edits)"],
["项目文件夹","Project folder"],["还没有选择","Not chosen yet"],["选择文件夹…","Choose folder…"],["取消选择","Clear"],
["或者粘贴文件夹路径，如 ~/Documents/myproject","Or paste a folder path, e.g. ~/Documents/myproject"],["文件夹路径","Folder path"],["使用这个路径","Use this path"],
["最近的修改","Recent changes"],["还没有修改。","No changes yet."],["（新文件）"," (new file)"],["已撤销。","Undone."],["已撤销","Undone"],["撤销","Undo"],
["请再选择一个项目文件夹。","Please also choose a project folder."],["请在弹出的窗口里选择文件夹…","Choose a folder in the window that appears…"],
["已设置项目文件夹，并开启了 Agent。","Project folder set and Agent turned on."],["已取消选择。","Selection cleared."],
["应用修改","Apply"],["拒绝","Reject"],["等待你的批准","Waiting for your approval"],["✓ 已应用","✓ Applied"],["已拒绝","Rejected"],["写入失败","Write failed"],
["Agent：开","Agent: on"],
// ---- Agent 服务端消息
["请先选择一个项目文件夹","Choose a project folder first"],["这个文件夹不存在","This folder does not exist"],
["为了安全，不能把根目录、用户主目录或系统目录当作项目文件夹，请选择具体的项目文件夹","For safety the root, home or system folders can't be used as a project folder; choose a specific project folder"],
["路径无效","Invalid path"],["路径不在项目文件夹内","The path is outside the project folder"],
["出于安全，这类文件（密钥、.env、.git 等）不允许读写","For safety, such files (keys, .env, .git, …) can't be read or written"],
["不存在：","Does not exist: "],["这不是文件夹","This is not a folder"],["这是文件夹，请用 list_dir","This is a folder; use list_dir"],
["文件太大（超过 1MB）","File too large (over 1MB)"],["这是二进制文件","This is a binary file"],["搜索内容是空的","The search text is empty"],
["old 不能为空；要新建或整体覆盖文件请用 write_file","old can't be empty; use write_file to create or overwrite a file"],
["在文件里找不到 old 这段文字（必须与文件内容完全一致，包括缩进）。请先用 read_file 看准再改","The old text was not found in the file (it must match exactly, including indentation). read_file first."],
["请多带几行上下文让它唯一","include more surrounding lines to make it unique"],["new 和 old 相同，没有改动","new equals old; nothing to change"],
["内容太大（超过 512KB）","Content too large (over 512KB)"],["内容与现有文件相同，没有改动","Same as the existing file; nothing to change"],
["找不到这条修改（可能已过期）","Change not found (it may have expired)"],["这条修改已经处理过了","This change has already been handled"],
["文件在你审批之前已经被创建了","The file was created before you approved"],["文件在提出修改之后又被改动过，请让模型重新读取后再改","The file changed after the edit was proposed; ask the model to read it again"],
["找不到这次修改的记录","No record of this change"],["这次修改已经撤销过了","This change was already undone"],
["文件在这次修改之后又被改动过，为避免覆盖你的内容，没有自动撤销","The file changed after this edit; not undone automatically to avoid overwriting your changes"],
["自动选择文件夹目前只支持 macOS，请直接粘贴文件夹路径","Folder picking is macOS-only for now; paste the folder path instead"],
["测试搜索","Test search"],["已保存。","Saved."],["正在测试…","Testing…"],["成功，第一条结果：","OK, first result: "],["(空)","(empty)"],
["读取文件失败","Could not read the file"],
["把书和文档导入到这里，对话时模型会先从书里找出相关段落再回答，并标出引用的是哪本书的第几段。全部在你自己的电脑上处理，不会上传。支持 txt / md / html / epub / docx / 代码文件；PDF 需要电脑里装有 pypdf 或 pdftotext，否则请先导出为 txt 或 epub。","Import books and documents here. When you chat, the model first finds the relevant passages and then answers, citing which book and passage it used. Everything is processed on your own computer and nothing is uploaded. Supports txt / md / html / epub / docx / code files; PDF needs pypdf or pdftotext installed, otherwise export it to txt or epub first."],
["聊天时使用知识库","Use the knowledge base in chat"],["（当前启用 ","(currently "],["选择要导入的书","Choose books to import"],
["万字"," ×10k chars"],["再点一次确认","Click again to confirm"],["删除","Delete"],["还没有导入任何书。","No books imported yet."],
["已开启知识库：回答前会先从你导入的书里查资料。","Knowledge base on: the model will look things up in your books before answering."],["已关闭知识库。","Knowledge base off."],["已删除。","Deleted."],
["正在导入 ","Importing "],["），大书需要一会儿…","), large books take a moment…"],
["导入完成：成功 ","Import done: "],["，失败 ",", failed "],["文件超过 55MB","File is over 55MB"],
["已清空（演示模式）。","Cleared (demo mode)."],["已清空。","Cleared."],
["演示模式：只切换了结构预览，回复仍是模拟的。","Demo mode: only the structure preview switched; replies are still simulated."],
["（首次约一分钟，会先卸载上一个模型）"," (about a minute the first time; the previous model is unloaded first)"],["正在加载 ","Loading "],
["已加载","loaded"],["（带适配器）"," (with adapter)"],["加载失败：","Load failed: "],["（演示）"," (demo)"],["· 未下载","· not downloaded"],
["选择一个模型…","Choose a model…"],["不使用适配器","No adapter"],["已连接。当前模型：","Connected. Current model: "],
["已连接本地服务。请在上方选择一个模型开始。","Connected to the local service. Choose a model above to start."],
["当前是演示模式：回复和活动均为模拟。启动本地服务后会自动连接真实模型。","Demo mode: replies and activity are simulated. Start the local service to connect a real model."],
["这个功能需要本地服务","This feature needs the local service"],
["现在看到的是演示页面，没有连接到你电脑上的 Neurolume。在项目文件夹里双击","This is a demo page, not connected to Neurolume on your computer. Double-click "],
["，浏览器会自动打开完整版本；之后下载模型、加载、训练都能在界面上点按钮完成。"," in the project folder; the full version opens in your browser, and downloading, loading and training models are all done with buttons."],
["添加其他模型：粘贴 Hugging Face 上的名字，如 mlx-community/…","Add another model: paste its Hugging Face name, e.g. mlx-community/…"],["模型名","Model name"],["检查并添加","Check and add"],
["想换模型：先“下载”，下载完点“加载”即可。已下载的模型保存在项目的 models 文件夹里，不用的可以删除来腾出空间。","To switch models: click “Download”, then “Load”. Downloaded models live in the project's models folder; delete the ones you don't use to free space."],
["正在检查…","Checking…"],["磁盘可用 ","Disk free "],["· 模型保存在 ","· models stored in "],["约 ","about "],
["已下载","Downloaded"],["下载中","Downloading"],["未下载","Not downloaded"],["取消下载","Cancel download"],["卸载","Unload"],["加载","Load"],
["生成中…","Generating…"],["生成权重图","Generate weight map"],["再点一次确认删除","Click again to delete"],["下载","Download"],["从列表移除","Remove from list"],["· 权重图已生成","· weight map ready"],
["下载失败：","Download failed: "],["生成权重图失败：","Weight map failed: "],["已删除，释放 ","Deleted, freed "],
["选数据","Data"],["选模型和强度","Model & intensity"],["完成","Done"],
["训练让 AI 学会你的写法和做事套路（例如你的代码风格、常用做法）。想让它记住书本里的知识，放进知识库更合适——这个功能稍后加入。","Training teaches the AI your writing and habits (e.g. your code style). To make it remember knowledge from books, use the knowledge base instead."],
["把文件拖到这里，或点击选择","Drop files here, or click to choose"],
["支持 .jsonl（对话/问答）、.md、.txt 和代码文件；含密钥的样本会被自动剔除","Supports .jsonl (chat / Q&A), .md, .txt and code files; samples containing secrets are removed automatically"],
["导入并检查","Import and check"],["或者使用已导入的数据","Or use data you already imported"],
["条训练 / "," train / "],["条验证"," valid"],["用这份","Use this"],["数据：","Data: "],
["条训练样本，"," training samples, "],["条用来检验"," for validation"],
["还没有已下载的模型。请先到“模型库”下载一个。","No downloaded models yet. Download one in “Models” first."],
["训练哪个模型","Which model to train"],["训练强度","Training intensity"],["快速试训","Quick trial"],["标准","Standard"],
[" 步 · 先确认流程能跑通、看看效果"," steps · check the pipeline works and see the effect"],[" 步 · 数据多时用"," steps · for larger data"],
["开始后会先卸载当前模型来腾出内存，训练期间聊天暂停。原模型文件不会被改动：训练结果是一个单独的“适配器”，不满意可以随时不用。","Starting unloads the current model to free memory, and chat pauses during training. The original model files are not modified: the result is a separate “adapter” you can stop using any time."],
["上一步","Back"],["开始训练","Start training"],["已准备好：","Ready: "],["条训练 + "," train + "],["条验证"," valid"],["下一步","Next"],
["分钟"," min"],["正在训练","Training"],["训练失败","Training failed"],["已取消","Cancelled"],
["进度","Progress"],["训练损失","Train loss"],["验证损失","Valid loss"],["速度","Speed"],["峰值内存","Peak memory"],["剩余时间","Time left"],
["损失（loss）越低，说明它越能复现你的数据。你可以关闭这个窗口，训练会在后台继续；主页的“训练”视图会显示正在被修改的层和权重。","Lower loss means it reproduces your data better. You can close this window; training continues in the background, and the “Train” view on the home page shows the layers and weights being changed."],
["取消训练","Cancel training"],["重新开始","Start over"],["训练完成","Training complete"],["损失 ","Loss "],["· 验证损失","· valid loss"],["· 用时 ","· took "],
["注意：新的适配器还没有经过自动评测，先小范围试用。效果不好就点“恢复原模型”，不会有任何损失。","Note: the new adapter has not been auto-evaluated yet, so try it on a small scale first. If it's not good, click “Restore original model” — nothing is lost."],
["试用这个适配器","Try this adapter"],["恢复原模型（不用适配器）","Restore original model (no adapter)"],["再训练一次","Train again"],
["适配器 ","Adapter "],
// ---- 服务端消息
["这个模型还没下载，请先在“模型库”里下载","This model isn't downloaded yet. Download it in “Models” first."],
["正在训练，训练期间无法加载模型","Training is running; models can't be loaded meanwhile"],
["正在训练，训练期间无法聊天","Training is running; chat is unavailable meanwhile"],
["正在训练，请等训练结束再下载","Training is running; wait until it finishes before downloading"],
["这个模型正在使用，请先卸载","This model is in use; unload it first"],
["请选择真实模型来训练","Choose a real model to train"],
["还没有加载模型，请先在上方选择一个","No model loaded yet; choose one above first"],
["没有找到：","Not found: "],["未知的搜索来源","Unknown search provider"],
["SearXNG 地址要以 http:// 或 https:// 开头","The SearXNG address must start with http:// or https://"],
["请输入 Hugging Face 上的模型名，格式如 mlx-community/某个模型","Enter a Hugging Face model name, like mlx-community/some-model"],
["这个模型已经在列表里了","This model is already in the list"],
["找不到这个模型，或网络不可用（","Model not found, or the network is unavailable ("],
["这个仓库不是可直接加载的模型（需要 config.json 和 .safetensors 权重）","This repo isn't a directly loadable model (needs config.json and .safetensors weights)"],
["磁盘空间不够：需要约 ","Not enough disk space: need about "],[" GB，可用 "," GB, available "],
["下载结束但文件不完整，请重试","Download finished but files are incomplete; please retry"],["正在下载，请先取消","Downloading; cancel first"],
["拒绝删除模型目录之外的路径","Refusing to delete paths outside the models folder"],
["只支持 http/https 网址","Only http/https URLs are supported"],["出于安全考虑，不读取本机或内网地址","For safety, local and internal addresses are not read"],
["不支持的内容类型：","Unsupported content type: "],["网站返回 ","The site returned "],["连接失败：","Connection failed: "],
["搜索词是空的","The search query is empty"],["还没有填写 SearXNG 地址","The SearXNG address is not set"],
["SearXNG 没有返回 JSON（需要在它的设置里启用 json 格式）","SearXNG did not return JSON (enable the json format in its settings)"],
["没有解析到搜索结果（DuckDuckGo 页面可能改版或要求验证）","No search results parsed (the DuckDuckGo page may have changed or asked for verification)"],
["数据太大（超过 50MB），请分批导入","Data too large (over 50MB); import in batches"],
["已剔除 ","Removed "],[" 条疑似含密钥/令牌的样本"," samples that look like they contain keys/tokens"],
[" 行格式无法识别，已跳过"," lines with unrecognised format skipped"],["去掉 ","Dropped "],[" 条重复样本"," duplicate samples"],
["样本少于 20 条，训练效果会很不明显；建议多导入一些","Fewer than 20 samples; the effect will be barely noticeable. Import more."],
["这是大段文档内容。想让 AI 记住书本知识，建议放进知识库（检索）而不是微调：微调更适合学习写法、风格和做事套路","This looks like long document content. To make the AI remember book knowledge, use the knowledge base (retrieval) rather than fine-tuning: fine-tuning suits learning style and habits."],
["没有读到可用的样本。支持：.jsonl（messages / prompt+completion / instruction+output / text）、.md、.txt 和代码文件","No usable samples found. Supported: .jsonl (messages / prompt+completion / instruction+output / text), .md, .txt and code files"],
["已有训练在进行","A training run is already in progress"],
["内存不足：请关闭其他大型应用，或选择更小的模型/更少的数据","Out of memory: close other large apps, or choose a smaller model / less data"],
["训练异常退出（代码 ","Training exited abnormally (code "],
["这个 epub 文件已损坏","This epub file is corrupted"],["这个 docx 文件无法读取","This docx file can't be read"],["PDF 读取失败：","PDF read failed: "],
["读取 PDF 需要 pypdf：在设置里让 Neurolume 安装，或先把 PDF 导出为 txt / epub 再导入","Reading PDF needs pypdf; install it, or export the PDF to txt / epub first"],
["不支持的文件格式（二进制）","Unsupported file format (binary)"],["这本书太大（超过 3000 万字），请拆分后导入","This book is too large (over 30M characters); split it and import again"],
["没有读到可用的文字内容","No usable text found"],["这本书已经导入过了","This book has already been imported"],["文件内容无法解码","The file content can't be decoded"],
["无效的书 ID","Invalid book ID"],["找不到这本书","Book not found"],
["正在搜索：","Searching: "],["没有找到结果","No results found"],
["搜索失败：","Search failed: "],
["《","“"],["》","”"],
// ---- 标点（最后统一处理）
];
const EN_RX=[
[/(\d+) 个字/g,"$1 tokens"],[/第 (\d+) 个：「/g,"token $1: “"],[/」/g,"”"],[/从第 (\d+) 层起就定下来了/g,"settled from layer $1 on"],
[/第 (\d+)\/(\d+) 段/g,"piece $1/$2"],[/已处理 (\d+)\/(\d+) 段，得到 (\d+) 条/g,"Processed $1/$2 pieces, $3 items so far"],[/共 (\d+) 条。/g,"$1 items. "],[/保留第 (\d+) 条/g,"Keep item $1"],
[/《([^》]*)》第 (\d+) 段/g,"“$1” passage $2"],[/（当前启用 (\d+) 段）/g,(m,n)=>" ("+n+" passage"+(n==1?"":"s")+" enabled)"],[/(\d+) 段/g,(m,n)=>n+" passage"+(n==1?"":"s")],[/([\d.]+) 万字/g,(m,n)=>Math.round(n*10)/10+"×10k chars"],[/^(\d) 训练$/,"$1 Train"],[/^(\d) 完成$/,"$1 Done"],[/(\d+) 时 /g,"$1 h "],[/(\d+) 分钟/g,"$1 min"],[/(\d+) 分(?!钟)/g,"$1 min"],
[/已读取 (\d+) 个网页，正在整理回答/g,"Read $1 pages, preparing the answer"],
[/第 (\d+) 段/g,"passage $1"],[/第 (\d+) 层 · /g,"Layer $1 · "],
[/(\d+) 条训练样本，(\d+) 条用来检验/g,"$1 training samples, $2 for validation"],[/(\d+) 条训练/g,"$1 train"],[/(\d+) 条验证/g,"$1 valid"],[/(\d+) 本/g,"$1 book(s)"],
[/\b(\d+) 层\b/g,"$1 layers"],
];

// 项目原名 Cortex：把旧的界面设置（localStorage 的 cortex.*）迁移到 neurolume.*
try{for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);if(k&&k.indexOf('cortex.')===0){const n='neurolume.'+k.slice(7);if(localStorage.getItem(n)==null)localStorage.setItem(n,localStorage.getItem(k))}}}catch(_){}
var LANG='en';try{const v=localStorage.getItem('neurolume.lang');LANG=v==='zh'?'zh':'en'}catch(_){}
const ZHRE=/[㐀-鿿]/,_ex=new Map(),_ph=[];
EN.forEach(([z,e])=>{if(/^[㐀-鿿]{1,2}$/.test(z))_ex.set(z,e);else _ph.push([z,e])});_ph.sort((a,b)=>b[0].length-a[0].length);
function tr(s){
  if(LANG!=='en'||!ZHRE.test(s))return s;
  const m=/^(\s*)([\s\S]*?)(\s*)$/.exec(s),t=m[2];if(_ex.has(t))return m[1]+_ex.get(t)+m[3];
  let o=t;for(const [r,e] of EN_RX)o=o.replace(r,e);
  for(const [z,e] of _ph)if(o.indexOf(z)>=0)o=o.split(z).join(e);
  o=o.replace(/：/g,': ').replace(/，/g,', ').replace(/。/g,'. ').replace(/；/g,'; ').replace(/、/g,', ').replace(/（/g,' (').replace(/）/g,')').replace(/[“”]/g,'"').replace(/ {2,}/g,' ').replace(/ ([,;:)])/g,'$1').replace(/([.!?])(?=[A-Z])/g,'$1 ').replace(/\. $/,'.');
  return m[1]+o+m[3]}
const _tn=new Map(),_ta=new Map();   // 记住原文，切回中文时还原
const _skip=(n,attr)=>{const e=n.nodeType===3?n.parentElement:n;return !e||!!e.closest('.m.u,.m.a,script,style,pre.pv,pre.diff,[data-notr]'+(attr?'':',textarea'))};
function trText(n){
  if(LANG!=='en'||_skip(n))return;const v=n.data,rec=_tn.get(n);
  if(rec&&rec.en===v)return;                                  // 已经是我们翻译过的英文
  if(!ZHRE.test(v))return;const en=tr(v);if(en!==v){_tn.set(n,{zh:v,en});n.data=en}}
const ATTR=['placeholder','title','aria-label'];
function trAttrs(el){
  if(LANG!=='en'||el.nodeType!==1||_skip(el,1))return;
  for(const a of ATTR){const v=el.getAttribute(a);if(v==null)continue;let r=_ta.get(el);if(r&&r[a]&&r[a].en===v)continue;if(!ZHRE.test(v))continue;
    const en=tr(v);if(en!==v){if(!r){r={};_ta.set(el,r)}r[a]={zh:v,en};el.setAttribute(a,en)}}}
function trTree(root){
  if(LANG!=='en'||!root)return;
  if(root.nodeType===3){trText(root);return}
  if(root.nodeType!==1)return;
  trAttrs(root);root.querySelectorAll('[placeholder],[title],[aria-label]').forEach(trAttrs);
  const w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let n;while((n=w.nextNode()))trText(n)}
function untr(){
  _tn.forEach((r,n)=>{if(n.data===r.en)n.data=r.zh});_tn.clear();
  _ta.forEach((r,el)=>{for(const a in r)if(el.getAttribute(a)===r[a].en)el.setAttribute(a,r[a].zh)});_ta.clear()}
let _lobs=null;
function applyLang(){
  document.documentElement.lang=LANG==='en'?'en':'zh-CN';
  document.title=LANG==='en'?'Neurolume Brain':'Neurolume 大脑';
  const b=document.getElementById('btnLang');if(b){b.textContent=LANG==='en'?'中文':'English';b.title=LANG==='en'?'切换为中文':'Switch to English'}
  if(LANG==='en')trTree(document.body);else untr();
  try{if(typeof setShape==='function'){const sb=document.getElementById('btnShape');if(sb)sb.textContent=SHAPE==='nebula'?(LANG==='en'?'Shape: Nebula':'形态：星云'):(LANG==='en'?'Shape: Sphere':'形态：球体')}}catch(_){}
  try{setBooks(BOOKS);setWeb(WEB)}catch(_){}}
function setLang(v){LANG=v;try{localStorage.setItem('neurolume.lang',v)}catch(_){}applyLang()}
function initLang(){
  if(typeof MutationObserver==='undefined')return;
  _lobs=new MutationObserver(ms=>{if(LANG!=='en')return;for(const m of ms){if(m.type==='characterData')trText(m.target);else if(m.type==='attributes')trAttrs(m.target);else m.addedNodes.forEach(trTree)}});
  _lobs.observe(document.body,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:ATTR});
  const b=document.getElementById('btnLang');if(b)b.addEventListener('click',()=>setLang(LANG==='en'?'zh':'en'));
  applyLang()}

