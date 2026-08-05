export class QZ {
    static #TypeId = {
        BOOL: 0,
        INT8_T: 1,
        INT16_T: 2,
        INT32_T: 3,
        INT64_T: 4,
        UINT8_T: 5,
        UINT16_T: 6,
        UINT32_T: 7,
        UINT64_T: 8,
        FLOAT: 9,
        DOUBLE: 10,
        LONG_DOUBLE: 11,
        CHAR: 12,
        STRING: 13,
        PTR: 14,
        ARRAY: 15,
        VECTOR: 16,
        LIST: 17,
        DEQUE: 18,
        MAP: 19,
        UNORDERED_MAP: 20,
        SET: 21,
        UNORDERED_SET: 22,
        OPTIONAL: 23,
        NAMED_STRUCT: 24
    }

    /**
     * @brief 序列化
     * @param {object} object 序列化的内容
     * @returns {Uint8Array}
     */
    static serialize(object) {
        return this.#serialize_any(object);
    }

    /**
     * @brief 反序列化
     * @param {string | Uint8Array} content 二进制流
     * @returns {object}
     */
    static deserialize(content) {
        if (typeof content == "string") {
            let encoder = new TextEncoder();
            content = encoder.encode(content);
        }
        const result = this.#deserialize_any(content, 0);
        return result.data;
    }

    /**
     * @brief 序列化任意类型
     * @param {object} object 序列化的内容
     * @returns {Uint8Array}
     */
    static #serialize_any(object) {
        if (typeof object === "boolean") {
            return this.#serialize_bool(object);
        }
        if (typeof object === "number") {
            if (Number.isInteger(object)) {
                return this.#serialize_int(object);
            }
            return this.#serialize_float(object);
        }
        if (typeof object === "bigint") {
            return this.#serialize_int(object);
        }
        if (typeof object === "string") {
            return this.#serialize_string(object);
        }
        if (object === null || object === undefined) {
            return this.#serialize_optional(object);
        }
        if (Array.isArray(object) || ArrayBuffer.isView(object) || object instanceof Set) {
            return this.#serialize_list(object);
        }
        if (object instanceof Map || typeof object == "object") {
            return this.#serialize_dict(object);
        }
        throw new Error(`Unsupported type: ${typeof object}`);
    }

    /**
     * @brief 序列化Bool
     * @param {boolean} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_bool(object) {
        const type_id = this.#serialize_type_id(this.#TypeId.BOOL);
        const byte = new Uint8Array([object ? 1 : 0]);
        const result = new Uint8Array(type_id.length + byte.length);
        result.set(type_id, 0);
        result.set(byte, type_id.length);
        return result;
    }

    /**
     * @brief 序列化int
     * @param {number|bigint} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_int(object) {
        let data = null;
        let type_id_val = 0;

        if (typeof object === 'bigint' || object > 4294967295 || object < -2147483648) {
             // INT64 / UINT64
             const isNeg = object < 0;
             type_id_val = isNeg ? this.#TypeId.INT64_T : this.#TypeId.UINT64_T;
             const type_id = this.#serialize_type_id(type_id_val);
             const buffer = new ArrayBuffer(9); // 1 byte type + 8 bytes data
             const view = new DataView(buffer);
             view.setUint8(0, type_id_val);
             if (isNeg) {
                 view.setBigInt64(1, BigInt(object), true);
             } else {
                 view.setBigUint64(1, BigInt(object), true);
             }
             data = new Uint8Array(buffer);
        } else if (object > 65535 || object < -32768) {
             // INT32 / UINT32
             const isNeg = object < 0;
             type_id_val = isNeg ? this.#TypeId.INT32_T : this.#TypeId.UINT32_T;
             const buffer = new ArrayBuffer(5); // 1 byte type + 4 bytes data
             const view = new DataView(buffer);
             view.setUint8(0, type_id_val);
             if (isNeg) {
                 view.setInt32(1, object, true);
             } else {
                 view.setUInt32(1, object, true);
             }
             data = new Uint8Array(buffer);
        } else if (object > 255 || object < -128) {
             // INT16 / UINT16
             const isNeg = object < 0;
             type_id_val = isNeg ? this.#TypeId.INT16_T : this.#TypeId.UINT16_T;
             const buffer = new ArrayBuffer(3); // 1 byte type + 2 bytes data
             const view = new DataView(buffer);
             view.setUint8(0, type_id_val);
             if (isNeg) {
                 view.setInt16(1, object, true);
             } else {
                 view.setUInt16(1, object, true);
             }
             data = new Uint8Array(buffer);
        } else {
             // INT8 / UINT8
             const isNeg = object < 0;
             type_id_val = isNeg ? this.#TypeId.INT8_T : this.#TypeId.UINT8_T;
             const buffer = new ArrayBuffer(2); // 1 byte type + 1 byte data
             const view = new DataView(buffer);
             view.setUint8(0, type_id_val);
             if (isNeg) {
                 view.setInt8(1, object);
             } else {
                 view.setUint8(1, object);
             }
             data = new Uint8Array(buffer);
        }

        return data;
    }

    /**
     * @brief 序列化float
     * @param {number} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_float(object) {
        const type_id_val = this.#TypeId.DOUBLE;
        const buffer = new ArrayBuffer(9); // 1 byte type + 8 bytes data
        const view = new DataView(buffer);
        view.setUint8(0, type_id_val);
        view.setFloat64(1, object, true);
        return new Uint8Array(buffer);
    }

    /**
     * @brief 序列化字符串
     * @param {string} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_string(object) {
        const type_id = this.#serialize_type_id(this.#TypeId.STRING);
        const encoder = new TextEncoder();
        const string_array = encoder.encode(object);
        
        // 先序列化长度
        const length_array = this.#serialize_int(string_array.length);
        
        const totalLength = type_id.length + length_array.length + string_array.length;
        const buffer = new ArrayBuffer(totalLength);
        const view = new Uint8Array(buffer);
        
        let offset = 0;
        view.set(type_id, offset);
        offset += type_id.length;
        
        view.set(length_array, offset);
        offset += length_array.length;
        
        view.set(string_array, offset);
        
        return view;
    }

    /**
     * @brief 序列化可空类型
     * @param {null} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_optional(object) {
        return this.#serialize_type_id(this.#TypeId.OPTIONAL);
    }

    /**
     * @brief 序列化List类型
     * @param {Array | ArrayBuffer | Set} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_list(object) {
        const type_id = this.#serialize_type_id(this.#TypeId.LIST);
        const elements = [];
        let data_length = 0;

        // 将 Set 转为 Array
        const arr = object instanceof Set ? Array.from(object) : object;

        for (const element of arr) {
            const element_array = this.#serialize_any(element);
            data_length += element_array.length;
            elements.push(element_array);
        }

        // 序列化列表长度
        const length_array = this.#serialize_int(arr.length);
        
        const totalLength = type_id.length + length_array.length + data_length;
        const buffer = new Uint8Array(totalLength);
        
        let offset = 0;
        buffer.set(type_id, offset);
        offset += type_id.length;
        
        buffer.set(length_array, offset);
        offset += length_array.length;
        
        for (const elem of elements) {
            buffer.set(elem, offset);
            offset += elem.length;
        }
        
        return buffer;
    }

    /**
     * @brief 序列化Dict类型
     * @param {object} object 对象
     * @returns {Uint8Array}
     */
    static #serialize_dict(object) {
        let entries = null;
        if (object instanceof Map) {
            entries = Array.from(object.entries());
        } else if (typeof object === "object" && object !== null) {
            entries = Object.entries(object);
        }
        
        if (!entries) {
            return this.#serialize_optional(null);
        }

        const type_id = this.#serialize_type_id(this.#TypeId.MAP);
        const length_array = this.#serialize_int(entries.length);

        const key_elements = [];
        const value_elements = [];
        let data_length = 0;

        for (const [key, value] of entries) {
            const key_array = this.#serialize_any(key);
            const value_array = this.#serialize_any(value);
            data_length += key_array.length + value_array.length;
            key_elements.push(key_array);
            value_elements.push(value_array);
        }

        const totalLength = type_id.length + length_array.length + data_length;
        const buffer = new Uint8Array(totalLength);
        
        let offset = 0;
        buffer.set(type_id, offset);
        offset += type_id.length;
        
        buffer.set(length_array, offset);
        offset += length_array.length;
        
        for (let i = 0; i < key_elements.length; ++i) {
            buffer.set(key_elements[i], offset);
            offset += key_elements[i].length;
            
            buffer.set(value_elements[i], offset);
            offset += value_elements[i].length;
        }

        return buffer;
    }

    /**
     * @brief 序列化类型ID
     * @param {number} typeId 类型ID
     * @returns {Uint8Array}
     */
    static #serialize_type_id(typeId) {
        return new Uint8Array([typeId]);
    }

    /**
     * @brief 反序列化任意类型
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: object, index: number}}
     */
    static #deserialize_any(content, index) {
        const type_id_obj = this.#deserialize_type_id(content, index);
        const typeId = type_id_obj.data;
        index = type_id_obj.index;

        switch (typeId) {
            case this.#TypeId.BOOL:
                return this.#deserialize_bool(content, index);
            case this.#TypeId.INT8_T:
            case this.#TypeId.INT16_T:
            case this.#TypeId.INT32_T:
            case this.#TypeId.INT64_T:
            case this.#TypeId.UINT8_T:
            case this.#TypeId.UINT16_T:
            case this.#TypeId.UINT32_T:
            case this.#TypeId.UINT64_T:
                return this.#deserialize_int(content, index, typeId);
            case this.#TypeId.FLOAT:
            case this.#TypeId.DOUBLE:
            case this.#TypeId.LONG_DOUBLE:
                return this.#deserialize_float(content, index, typeId);
            case this.#TypeId.CHAR:
            case this.#TypeId.STRING:
                return this.#deserialize_string(content, index);
            case this.#TypeId.ARRAY:
            case this.#TypeId.VECTOR:
            case this.#TypeId.LIST:
            case this.#TypeId.DEQUE:
            case this.#TypeId.SET:
            case this.#TypeId.UNORDERED_SET:
                return this.#deserialize_list(content, index);
            case this.#TypeId.MAP:
            case this.#TypeId.UNORDERED_MAP:
            case this.#TypeId.NAMED_STRUCT:
                return this.#deserialize_dict(content, index);
            case this.#TypeId.PTR:
            case this.#TypeId.OPTIONAL:
                return this.#deserialize_optional(content, index);
            default:
                throw new Error(`Unknown type ID: ${typeId} at index ${index - 1}`);
        }
    }

    /**
     * @brief 反序列化Bool
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: bool, index: number}}
     */
    static #deserialize_bool(content, index) {
        const data_view = new DataView(content.buffer, content.byteOffset + index);
        const val = data_view.getInt8(0);
        return {
            data: val != 0,
            index: index + 1
        };
    }

    /**
     * @brief 反序列化int
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @param {number} typeId 已经读取到的类型ID
     * @returns {{data: number, index: number}}
     */
    static #deserialize_int(content, index, typeId) {
        const data_view = new DataView(content.buffer, content.byteOffset + index);
        let data = 0;
        let byteSize = 0;

        switch (typeId) {
            case this.#TypeId.INT8_T:
                data = data_view.getInt8(0);
                byteSize = 1;
                break;
            case this.#TypeId.INT16_T:
                data = data_view.getInt16(0, true);
                byteSize = 2;
                break;
            case this.#TypeId.INT32_T:
                data = data_view.getInt32(0, true);
                byteSize = 4;
                break;
            case this.#TypeId.INT64_T:
                data = Number(data_view.getBigInt64(0, true));
                byteSize = 8;
                break;
            case this.#TypeId.UINT8_T:
                data = data_view.getUint8(0);
                byteSize = 1;
                break;
            case this.#TypeId.UINT16_T:
                data = data_view.getUint16(0, true);
                byteSize = 2;
                break;
            case this.#TypeId.UINT32_T:
                data = data_view.getUint32(0, true);
                byteSize = 4;
                break;
            case this.#TypeId.UINT64_T:
                data = Number(data_view.getBigUint64(0, true));
                byteSize = 8;
                break;
            default:
                throw new Error(`Type is not point to a int type. Got ID: ${typeId}`);
        }
        return {
            data: data,
            index: index + byteSize
        };
    }

    /**
     * @brief 反序列化float
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @param {number} typeId 已经读取到的类型ID
     * @returns {{data: number, index: number}}
     */
    static #deserialize_float(content, index, typeId) {
        const data_view = new DataView(content.buffer, content.byteOffset + index);
        let data = 0;
        let byteSize = 0;

        switch (typeId) {
            case this.#TypeId.FLOAT:
                data = data_view.getFloat32(0, true);
                byteSize = 4;
                break;
            case this.#TypeId.DOUBLE:
            case this.#TypeId.LONG_DOUBLE:
                data = data_view.getFloat64(0, true);
                byteSize = 8;
                break;
            default:
                throw new Error(`Type is not point to a float type. Got ID: ${typeId}`);
        }
        return {
            data: data,
            index: index + byteSize
        };
    }

    /**
     * @brief 反序列化字符串
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: string, index: number}}
     */
    static #deserialize_string(content, index) {
        const len_result = this.#deserialize_any(content, index);
        const str_len = len_result.data;
        let current_index = len_result.index;
        
        const decoder = new TextDecoder();

        if (current_index + str_len > content.length) {
             throw new Error("String length exceeds buffer bounds");
        }
        const data = decoder.decode(content.slice(current_index, current_index + str_len));
        
        return {
            data: data,
            index: current_index + str_len
        };
    }

    /**
     * @brief 反序列化可空类型
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: null, index: number}}
     */
    static #deserialize_optional(content, index) {
        return {
            data: null,
            index: index
        };
    }

    /**
     * @brief 反序列化List类型
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: object[], index: number}}
     */
    static #deserialize_list(content, index) {
        // 读取列表长度
        const length_obj = this.#deserialize_any(content, index);
        const list_len = length_obj.data;
        index = length_obj.index;

        const data = [];

        for (let i = 0; i < list_len; ++i) {
            const element_obj = this.#deserialize_any(content, index);
            data.push(element_obj.data);
            index = element_obj.index;
        }

        return {
            data: data,
            index: index
        };
    }

    /**
     * @brief 反序列化Dict类型
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: object, index: number}}
     */
    static #deserialize_dict(content, index) {
        // 读取字典长度
        const length_obj = this.#deserialize_any(content, index);
        const dict_len = length_obj.data;
        index = length_obj.index;

        const data = {};

        for (let i = 0; i < dict_len; ++i) {
            const key_obj = this.#deserialize_any(content, index);
            index = key_obj.index;
            
            const value_obj = this.#deserialize_any(content, index);
            index = value_obj.index;
            
            // JS 对象键必须是字符串或 Symbol，如果 key 是数字，这里会自动转换
            data[key_obj.data] = value_obj.data;
        }

        return {
            data: data,
            index: index
        };
    }

    /**
     * @brief 反序列化类型ID
     * @param {Uint8Array} content 二进制流
     * @param {number} index 读取指针
     * @returns {{data: int, index: number}}
     */
    static #deserialize_type_id(content, index) {
        if (index >= content.length) {
            throw new Error("Index out of bounds when reading type ID");
        }
        const data = content[index];
        return {
            data: data,
            index: index + 1
        };
    }
}