#pragma once

#include <string>

class Shader {
public:
    Shader(const std::string& name, const std::string& vertex_code, const std::string& fragment_code);
    ~Shader();

    bool compile();
    void use();
    void setUniform(const std::string& name, float value);
    void setUniform(const std::string& name, int value);
    void setUniform(const std::string& name, float x, float y, float z);

    const std::string& getName() const { return name_; }
    bool isCompiled() const { return compiled_; }

private:
    std::string name_;
    std::string vertex_shader_code_;
    std::string fragment_shader_code_;
    
    unsigned int program_id_;
    unsigned int vertex_shader_id_;
    unsigned int fragment_shader_id_;
    
    bool compiled_;
    
    bool compileShader(unsigned int& shader_id, const std::string& code, int type);
    void checkCompileErrors(unsigned int shader_id, int type);
};