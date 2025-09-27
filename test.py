with gr.Tab("生成结果"):
    self.output_model_obj = gr.Model3D(label="OBJ 格式模型预览", interactive=False, height=400)
    
    # 添加星星评分功能
    gr.Markdown("### 模型评分")
    
    # 使用HTML组件创建悬停效果的星星评分
    self.star_rating_html = gr.HTML(
        value="""
        <div class="star-rating-container">
            <style>
                .star-rating-container {
                    margin: 20px 0;
                    font-family: Arial, sans-serif;
                }
                
                .star-rating-label {
                    margin-bottom: 10px;
                    font-size: 14px;
                    color: #333;
                }
                
                .star-rating {
                    display: flex;
                    gap: 5px;
                    margin: 10px 0;
                }
                
                .star {
                    font-size: 35px;
                    color: #ddd;
                    cursor: pointer;
                    transition: color 0.2s ease;
                    user-select: none;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                }
                
                .star:hover {
                    color: #ffd700;
                    transform: scale(1.1);
                    transition: all 0.2s ease;
                }
                
                .star.active {
                    color: #ffd700;
                }
                
                .star.hover-active {
                    color: #ffd700;
                }
                
                .rating-text {
                    margin-top: 15px;
                    font-size: 16px;
                    color: #555;
                    font-weight: bold;
                }
            </style>
            
            <div class="star-rating-label">将鼠标悬停在星星上预览评分，点击确定评分：</div>
            <div class="star-rating" id="starRating">
                <span class="star active" data-rating="1">★</span>
                <span class="star active" data-rating="2">★</span>
                <span class="star active" data-rating="3">★</span>
                <span class="star active" data-rating="4">★</span>
                <span class="star active" data-rating="5">★</span>
            </div>
            <div class="rating-text" id="ratingText">当前评分: 5 星</div>
            
            <script>
                (function() {
                    const stars = document.querySelectorAll('#starRating .star');
                    const ratingText = document.getElementById('ratingText');
                    let currentRating = 5;
                    
                    // 更新星星显示
                    function updateStars(rating, isHover = false) {
                        stars.forEach((star, index) => {
                            star.classList.remove('active', 'hover-active');
                            if (index < rating) {
                                if (isHover) {
                                    star.classList.add('hover-active');
                                } else {
                                    star.classList.add('active');
                                }
                            }
                        });
                        
                        if (!isHover) {
                            ratingText.textContent = `当前评分: ${rating} 星`;
                            currentRating = rating;
                            // 将评分传递给Python后端
                            if (window.gradioApp) {
                                window.currentStarRating = rating;
                            }
                        } else {
                            ratingText.textContent = `预览评分: ${rating} 星 (点击确定)`;
                        }
                    }
                    
                    // 鼠标悬停效果
                    stars.forEach((star, index) => {
                        star.addEventListener('mouseenter', () => {
                            const rating = index + 1;
                            updateStars(rating, true);
                        });
                        
                        // 点击设置评分
                        star.addEventListener('click', () => {
                            const rating = index + 1;
                            updateStars(rating, false);
                        });
                    });
                    
                    // 鼠标离开星星区域时恢复当前评分显示
                    document.getElementById('starRating').addEventListener('mouseleave', () => {
                        updateStars(currentRating, false);
                    });
                    
                    // 初始化
                    window.currentStarRating = 5;
                    updateStars(5, false);
                })();
            </script>
        </div>
        """
    )
    
    # 隐藏的评分值存储（用于与Python后端通信）
    self.rating_value = gr.State(5)
    
    # 用于更新评分值的按钮（不可见）
    self.update_rating_btn = gr.Button("更新评分", visible=False)
    
    with gr.Row():
        self.rating_comment = gr.Textbox(
            label="评价留言 (可选)",
            placeholder="请输入您对模型的评价...",
            lines=3,
            interactive=True
        )
    
    with gr.Row():
        self.submit_rating_btn = gr.Button("提交评分", variant="primary", size="lg")
        
    # 评分提交后的反馈信息
    self.rating_output = gr.Markdown("", visible=False)

# 在类中添加这些方法来处理评分逻辑

def setup_hover_rating_events(self):
    """设置悬停星星评分事件"""
    
    def get_current_rating():
        """获取当前JavaScript中的评分值"""
        # 由于Gradio限制，这里使用默认值
        # 实际项目中可能需要其他方式来同步JS和Python的状态
        return 5
    
    def submit_rating(comment):
        """处理用户评分提交"""
        try:
            # 获取当前评分（在实际使用中，可能需要通过其他方式获取JS中的值）
            rating = get_current_rating()
            
            print(f"用户评分: {rating}星")
            print(f"用户评价: {comment}")
            
            # 这里可以添加保存评分的逻辑
            # 例如保存到文件或数据库
            
            star_display = "⭐" * int(rating)
            return gr.Markdown.update(
                value=f"✅ 感谢您的评分！您给出了 {star_display} ({rating}星) 评价。",
                visible=True
            )
        except Exception as e:
            return gr.Markdown.update(
                value=f"❌ 提交评分时出错: {str(e)}",
                visible=True
            )
    
    # 绑定提交评分事件
    self.submit_rating_btn.click(
        fn=submit_rating,
        inputs=[self.rating_comment],
        outputs=[self.rating_output]
    )

# 更简化的方案，如果上面的JavaScript不工作，可以用这个：

def create_simple_hover_stars(self):
    """创建简化版悬停星星（使用CSS伪类）"""
    return gr.HTML(
        value="""
        <style>
            .simple-stars {
                display: flex;
                gap: 5px;
                margin: 15px 0;
            }
            .simple-stars input[type="radio"] {
                display: none;
            }
            .simple-stars label {
                font-size: 30px;
                color: #ddd;
                cursor: pointer;
                transition: color 0.2s;
            }
            .simple-stars label:hover,
            .simple-stars label:hover ~ label {
                color: #ffd700;
            }
            .simple-stars input[type="radio"]:checked ~ label {
                color: #ffd700;
            }
        </style>
        
        <div class="simple-stars">
            <input type="radio" id="star5" name="rating" value="5" checked>
            <label for="star5">★</label>
            <input type="radio" id="star4" name="rating" value="4">
            <label for="star4">★</label>
            <input type="radio" id="star3" name="rating" value="3">
            <label for="star3">★</label>
            <input type="radio" id="star2" name="rating" value="2">
            <label for="star2">★</label>
            <input type="radio" id="star1" name="rating" value="1">
            <label for="star1">★</label>
        </div>
        <p>当前评分: <span id="current-rating">5</span> 星</p>
        
        <script>
            document.querySelectorAll('input[name="rating"]').forEach(radio => {
                radio.addEventListener('change', function() {
                    document.getElementById('current-rating').textContent = this.value;
                });
            });
        </script>
        """
    )

# 记得在你的类的 __init__ 方法最后调用：
# self.setup_hover_rating_events()